#!/usr/bin/env python3
"""
InsertSplits DAG Visualizer
===========================
Parses Caffe prototxt files, simulates the InsertSplits graph transformation
(two-pass algorithm), and visualizes the DAG before/after split insertion.

Supports both:
  - caffe-ffi variant: handles `input: "data"` (external inputs) with auto-split
  - native Caffe: external inputs come from type:"Input" layers

No protobuf dependency required — parses text format directly.

Usage:
  python viz_insert_splits.py                           # run built-in test cases
  python viz_insert_splits.py <file.prototxt>           # visualize a single file
  python viz_insert_splits.py <file.prototxt> --dot     # also output Graphviz DOT
  python viz_insert_splits.py --all                     # run all built-in cases
  python viz_insert_splits.py --case TwoConsumer        # run specific test case
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Layer:
    name: str
    type: str
    bottoms: list[str] = field(default_factory=list)
    tops: list[str] = field(default_factory=list)
    loss_weights: list[float] = field(default_factory=list)
    is_auto_split: bool = False  # true if inserted by our simulation

    def __repr__(self):
        b = ", ".join(self.bottoms) if self.bottoms else "-"
        t = ", ".join(self.tops) if self.tops else "-"
        return f"Layer({self.name!r}[{self.type}] bot=[{b}] top=[{t}])"


@dataclass
class NetSpec:
    name: str = ""
    inputs: list[str] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)

    def clone(self) -> NetSpec:
        return NetSpec(
            name=self.name,
            inputs=list(self.inputs),
            layers=[Layer(l.name, l.type, list(l.bottoms), list(l.tops),
                          list(l.loss_weights), l.is_auto_split) for l in self.layers],
        )


# ──────────────────────────────────────────────────────────────────────
# Prototxt parser (minimal — only extracts fields we need)
# ──────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[tuple[str, str]]:
    """Tokenize prototxt into (type, value) pairs.
    Types:
      'str'  — quoted string literal ("foo" or 'bar')
      'num'  — numeric literal (int/float)
      'ident'— unquoted identifier (field names, enum values, bare words)
      '{'    — open brace
      '}'    — close brace
    In protobuf text format, all string values are quoted; bare unquoted
    words are either field names (before ':' or '{') or enum values.
    """
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in ' \t\r\n':
            i += 1
            continue
        if c == '#':
            while i < n and text[i] != '\n':
                i += 1
            continue
        if c == '{':
            tokens.append(('{', '{'))
            i += 1
            continue
        if c == '}':
            tokens.append(('}', '}'))
            i += 1
            continue
        if c == '"' or c == "'":
            quote = c
            i += 1
            start = i
            while i < n and text[i] != quote:
                if text[i] == '\\':
                    i += 1
                i += 1
            val = text[start:i]
            i += 1  # skip closing quote
            tokens.append(('str', val))
            continue
        # Read a bare token (letters, digits, underscores, colons, dots, signs, slashes)
        start = i
        while i < n and text[i] not in ' \t\r\n{}#"\'':
            i += 1
        val = text[start:i]

        if ':' in val and not val.endswith(':'):
            # key:value on same token (e.g. dim:2 or num_output:3)
            parts = val.split(':', 1)
            tokens.append(('ident', parts[0]))
            v = parts[1].strip()
            if v:
                # try number first, then treat as ident (enum)
                try:
                    float(v)
                    tokens.append(('num', v))
                except ValueError:
                    tokens.append(('ident', v))
            continue
        if val.endswith(':'):
            # key:  (key followed by colon, value on next tokens)
            tokens.append(('ident', val[:-1]))
            continue
        # bare value — number or identifier
        try:
            float(val)
            tokens.append(('num', val))
        except ValueError:
            tokens.append(('ident', val))
    return tokens


def _skip_block_at(tokens, pos):
    """Skip a { ... } block starting at pos (pointing to '{'). Returns pos after '}'."""
    depth = 1
    pos += 1
    while pos < len(tokens) and depth > 0:
        if tokens[pos][0] == '{':
            depth += 1
        elif tokens[pos][0] == '}':
            depth -= 1
        pos += 1
    return pos


def parse_prototxt(text: str) -> NetSpec:
    """Parse Caffe prototxt text into a NetSpec.
    Only extracts: name, input, layer{name,type,bottom,top,loss_weight}.
    Inner param blocks (e.g. inner_product_param{...}) are skipped.
    """
    tokens = _tokenize(text)
    net = NetSpec()
    i = 0
    while i < len(tokens):
        ttype, tval = tokens[i]
        if ttype == 'ident' and tval == 'name':
            if i + 1 < len(tokens) and tokens[i + 1][0] == 'str':
                net.name = tokens[i + 1][1]
                i += 2
                continue
        if ttype == 'ident' and tval == 'input':
            if i + 1 < len(tokens) and tokens[i + 1][0] == 'str':
                net.inputs.append(tokens[i + 1][1])
                i += 2
                continue
        if ttype == 'ident' and tval == 'layer':
            layer = Layer(name="", type="")
            i += 1
            if i < len(tokens) and tokens[i][0] == '{':
                depth = 1
                i += 1
                while i < len(tokens) and depth > 0:
                    lt, lv = tokens[i]
                    if lt == '{':
                        i = _skip_block_at(tokens, i)
                        continue
                    if lt == '}':
                        depth -= 1
                        i += 1
                        continue
                    if lt == 'ident' and lv == 'name':
                        if i + 1 < len(tokens) and tokens[i + 1][0] == 'str':
                            layer.name = tokens[i + 1][1]
                            i += 2
                            continue
                    if lt == 'ident' and lv == 'type':
                        if i + 1 < len(tokens) and tokens[i + 1][0] == 'str':
                            layer.type = tokens[i + 1][1]
                            i += 2
                            continue
                    if lt == 'ident' and lv == 'bottom':
                        if i + 1 < len(tokens) and tokens[i + 1][0] == 'str':
                            layer.bottoms.append(tokens[i + 1][1])
                            i += 2
                            continue
                    if lt == 'ident' and lv == 'top':
                        if i + 1 < len(tokens) and tokens[i + 1][0] == 'str':
                            layer.tops.append(tokens[i + 1][1])
                            i += 2
                            continue
                    if lt == 'ident' and lv == 'loss_weight':
                        if i + 1 < len(tokens) and tokens[i + 1][0] == 'num':
                            layer.loss_weights.append(float(tokens[i + 1][1]))
                            i += 2
                            continue
                    i += 1
            net.layers.append(layer)
            continue
        if ttype == '{':
            i = _skip_block_at(tokens, i)
            continue
        i += 1
    return net


# ──────────────────────────────────────────────────────────────────────
# InsertSplits simulation (faithful to caffe-ffi/net.cpp implementation)
# ──────────────────────────────────────────────────────────────────────

def _split_layer_name(producer: str, blob: str, blob_idx: int) -> str:
    return f"{blob}_{producer}_{blob_idx}_split"


def _split_blob_name(producer: str, blob: str, blob_idx: int, split_idx: int) -> str:
    return f"{blob}_{producer}_{blob_idx}_split_{split_idx}"


def _make_split_layer(producer: str, blob: str, blob_idx: int,
                      split_count: int, loss_weight: float) -> Layer:
    """Create a Split layer (equivalent to ConfigureSplitLayer in C++)."""
    tops = [_split_blob_name(producer, blob, blob_idx, k) for k in range(split_count)]
    lws = []
    if loss_weight != 0.0:
        lws = [loss_weight if k == 0 else 0.0 for k in range(split_count)]
    return Layer(
        name=_split_layer_name(producer, blob, blob_idx),
        type="Split",
        bottoms=[blob],
        tops=tops,
        loss_weights=lws,
        is_auto_split=True,
    )


@dataclass
class FanoutAnalysis:
    """Result of Pass 1: consumer counts with correct in-place tracking."""
    # (layer_idx, top_idx) -> count of consumers; (-1,i) = external input i
    top_to_bottom_count: dict[tuple[int, int], int]
    # (layer_idx, top_idx) -> loss weight
    top_to_loss_weight: dict[tuple[int, int], float]
    # (consumer_layer_idx, bottom_idx) -> (producer_layer_idx, top_idx)
    bottom_to_source: dict[tuple[int, int], tuple[int, int]]
    # layer_idx -> layer_name
    layer_names: dict[int, str]
    # list of (producer_layer_idx, top_idx, blob_name, consumer_count, needs_split)
    blob_records: list[tuple]


def analyze_fanout(net: NetSpec, mode: str = "caffe-ffi") -> FanoutAnalysis:
    """Run Pass 1 of InsertSplits to compute accurate consumer counts
    (correctly handling in-place layers that update last-producer)."""
    blob_to_last_top: dict[str, tuple[int, int]] = {}
    bottom_to_source: dict[tuple[int, int], tuple[int, int]] = {}
    top_to_bottom_count: dict[tuple[int, int], int] = {}
    top_to_loss_weight: dict[tuple[int, int], float] = {}
    layer_names: dict[int, str] = {}

    # Register external inputs as virtual producers at (-1, i)
    if mode == "caffe-ffi":
        for idx, blob_name in enumerate(net.inputs):
            blob_to_last_top[blob_name] = (-1, idx)
            top_to_bottom_count[(-1, idx)] = 0

    # Pass 1: Count consumers
    for i, layer in enumerate(net.layers):
        layer_names[i] = layer.name
        for j, blob_name in enumerate(layer.bottoms):
            if blob_name not in blob_to_last_top:
                raise ValueError(
                    f"Unknown bottom blob '{blob_name}' "
                    f"(layer '{layer.name}', bottom index {j})"
                )
            source = blob_to_last_top[blob_name]
            bottom_to_source[(i, j)] = source
            top_to_bottom_count[source] = top_to_bottom_count.get(source, 0) + 1
        for j, blob_name in enumerate(layer.tops):
            blob_to_last_top[blob_name] = (i, j)
            if (i, j) not in top_to_bottom_count:
                top_to_bottom_count[(i, j)] = 0
        last_loss = min(len(layer.loss_weights), len(layer.tops))
        for j in range(last_loss):
            blob_name = layer.tops[j]
            lw = layer.loss_weights[j]
            top_idx = blob_to_last_top[blob_name]
            top_to_loss_weight[top_idx] = lw
            if lw != 0.0:
                top_to_bottom_count[top_idx] = top_to_bottom_count.get(top_idx, 0) + 1

    # Build blob records
    blob_records = []
    # External inputs
    if mode == "caffe-ffi":
        for idx, blob_name in enumerate(net.inputs):
            t_idx = (-1, idx)
            cnt = top_to_bottom_count.get(t_idx, 0)
            blob_records.append((-1, idx, blob_name, cnt, cnt > 1))
    # Layer tops
    for i, layer in enumerate(net.layers):
        for j, blob_name in enumerate(layer.tops):
            t_idx = (i, j)
            cnt = top_to_bottom_count.get(t_idx, 0)
            blob_records.append((i, j, blob_name, cnt, cnt > 1))

    return FanoutAnalysis(
        top_to_bottom_count=top_to_bottom_count,
        top_to_loss_weight=top_to_loss_weight,
        bottom_to_source=bottom_to_source,
        layer_names=layer_names,
        blob_records=blob_records,
    )


def simulate_insert_splits(net: NetSpec, mode: str = "caffe-ffi") -> NetSpec:
    """Simulate InsertSplits and return the transformed network.

    mode:
      'caffe-ffi' — handles param.input() external inputs directly (producer='input')
      'native'    — native Caffe behavior (external inputs should be Input layers)
    """
    out = net.clone()
    out.layers = []

    # Run Pass 1
    try:
        fa = analyze_fanout(net, mode)
    except ValueError:
        raise

    top_to_bottom_count = dict(fa.top_to_bottom_count)
    top_to_loss_weight = dict(fa.top_to_loss_weight)
    bottom_to_source = dict(fa.bottom_to_source)
    layer_names = dict(fa.layer_names)
    top_to_split_idx: dict[tuple[int, int], int] = {}

    # Check if any splits needed
    split_needed = sum(1 for c in top_to_bottom_count.values() if c > 1)
    if split_needed == 0:
        for layer in net.layers:
            out.layers.append(Layer(layer.name, layer.type, list(layer.bottoms),
                                    list(layer.tops), list(layer.loss_weights)))
        return out

    # === Pass 2: Rewrite bottoms and insert splits ===
    for i, layer in enumerate(net.layers):
        # Copy layer
        new_layer = Layer(layer.name, layer.type, list(layer.bottoms),
                          list(layer.tops), list(layer.loss_weights))
        out.layers.append(new_layer)

        # Step 2a: Rewrite bottoms for multi-consumer blobs
        for j in range(len(new_layer.bottoms)):
            source = bottom_to_source.get((i, j))
            if source is None:
                continue
            sc = top_to_bottom_count.get(source, 0)
            if sc > 1:
                src_layer, src_top = source
                blob_name = new_layer.bottoms[j]
                if src_layer == -1:
                    producer = "input"
                    ext_blob = net.inputs[src_top] if mode == "caffe-ffi" else blob_name
                    blob_name_for_split = ext_blob
                else:
                    producer = layer_names[src_layer]
                    blob_name_for_split = blob_name
                sidx = top_to_split_idx.get(source, 0)
                new_name = _split_blob_name(producer, blob_name_for_split, src_top, sidx)
                new_layer.bottoms[j] = new_name
                top_to_split_idx[source] = sidx + 1

        # Step 2b: After this layer, insert Split for tops that need it
        split_inserted_after = []
        for j in range(len(new_layer.tops)):
            top_idx = (i, j)
            sc = top_to_bottom_count.get(top_idx, 0)
            if sc > 1:
                producer = layer_names[i]
                blob_name = new_layer.tops[j]
                lw = top_to_loss_weight.get(top_idx, 0.0)
                split_layer = _make_split_layer(producer, blob_name, j, sc, lw)
                split_inserted_after.append(split_layer)
                if lw != 0.0:
                    # Clear loss weight on original layer
                    new_layer.loss_weights = [0.0] * len(new_layer.loss_weights)
                    top_to_split_idx[top_idx] = top_to_split_idx.get(top_idx, 0) + 1

        for sl in split_inserted_after:
            out.layers.append(sl)

    # === Pass 2b: Handle external input splits (insert at position 0) ===
    if mode == "caffe-ffi":
        input_splits = []
        for idx, blob_name in enumerate(net.inputs):
            top_idx = (-1, idx)
            sc = top_to_bottom_count.get(top_idx, 0)
            if sc > 1:
                sl = _make_split_layer("input", blob_name, idx, sc, 0.0)
                input_splits.append(sl)

        if input_splits:
            # Shift existing layers right, insert splits at front
            out.layers = input_splits + out.layers

    # For native Caffe mode, Input layers already produce splits in Pass 2b
    # (they are regular layers, splits inserted after them)

    return out


# ──────────────────────────────────────────────────────────────────────
# DAG visualization
# ──────────────────────────────────────────────────────────────────────

def _consumer_map(net: NetSpec, mode: str = "caffe-ffi") -> dict[str, list[tuple[str, str]]]:
    """Build {blob_name: [(consumer_layer_name, via_bottom)]} for visualization."""
    consumers: dict[str, list[tuple[str, str]]] = {}
    # Register external inputs
    if mode == "caffe-ffi":
        for inp in net.inputs:
            consumers[inp] = []
    for layer in net.layers:
        for bot in layer.bottoms:
            consumers.setdefault(bot, [])
            consumers[bot].append((layer.name, "bottom"))
        for top in layer.tops:
            consumers.setdefault(top, [])
    return consumers


def _producer_map(net: NetSpec, mode: str = "caffe-ffi") -> dict[str, str]:
    """Build {blob_name: producer_layer_name}."""
    producers: dict[str, str] = {}
    if mode == "caffe-ffi":
        for inp in net.inputs:
            producers[inp] = "<external>"
    for layer in net.layers:
        for top in layer.tops:
            producers[top] = layer.name
    return producers


def print_dag_table(net: NetSpec, title: str = "DAG", mode: str = "caffe-ffi",
                    show_warnings: bool = True, fanout: Optional[FanoutAnalysis] = None):
    """Print a tabular view of layers and blob fan-out.

    When show_warnings=True (before-view), marks blobs with fan-out > 1.
    When show_warnings=False (after-view), just shows the structure with
    auto-inserted splits highlighted.

    If fanout is provided (from analyze_fanout), uses accurate Pass 1 data
    for warnings (correctly handles in-place last-producer tracking).
    """
    w = 120
    print("=" * w)
    print(f"  {title}")
    print(f"  Net: {net.name or '(unnamed)'}  |  Layers: {len(net.layers)}  |  "
          f"External inputs: {len(net.inputs)}  |  Mode: {mode}")
    print("=" * w)

    def _trunc(s: str, maxlen: int) -> str:
        return s if len(s) <= maxlen else s[:maxlen - 1] + "…"

    # Build per-top fanout info from FanoutAnalysis if available
    # Maps (layer_idx, top_idx) -> (count, [consumer_names])
    top_fanout: dict[tuple[int, int], tuple[int, list[str]]] = {}
    ext_fanout: dict[int, tuple[int, list[str]]] = {}  # ext input idx -> (count, [names])
    if fanout is not None:
        src_to_cons: dict[tuple[int, int], list[str]] = {}
        for (ci, bj), (si, sj) in fanout.bottom_to_source.items():
            src_to_cons.setdefault((si, sj), []).append(
                fanout.layer_names.get(ci, f"?")
            )
        for (si, sj), cnt in fanout.top_to_bottom_count.items():
            cons = src_to_cons.get((si, sj), [])
            if si == -1:
                ext_fanout[sj] = (cnt, cons)
            else:
                top_fanout[(si, sj)] = (cnt, cons)
    else:
        # Fallback: simple consumer map (doesn't handle in-place correctly)
        consumers = _consumer_map(net, mode)

    # Header
    print(f"{'#':>4} {'Layer Name':<32} {'Type':<14} {'In-Blobs':<30} {'Out-Blobs':<30}")
    print("-" * w)

    pos = 0
    if mode == "caffe-ffi" and net.inputs:
        for idx, inp in enumerate(net.inputs):
            if fanout is not None:
                cnt, cons = ext_fanout.get(idx, (0, []))
            else:
                cons = [c[0] for c in consumers.get(inp, [])]
                cnt = len(cons)
            if show_warnings:
                marker = " ***" if cnt > 1 else "    "
            else:
                marker = "    "
            print(f"{marker}{pos:>3} {'<external input>':<32} {'INPUT':<14} "
                  f"{'-':<30} {inp:<30}")
            if show_warnings and cnt > 1:
                print(f"{'':>5}  ⚠ fan-out={cnt}: {', '.join(cons)}")
            elif show_warnings:
                c_str = f"→ {cons[0]}" if cons else "(unused)"
                print(f"{'':>5}  consumers: {c_str}")
            pos += 1

    for i, layer in enumerate(net.layers):
        bot_str = ", ".join(layer.bottoms) if layer.bottoms else "-"
        top_str = ", ".join(layer.tops) if layer.tops else "-"
        bot_str = _trunc(bot_str, 28)
        top_str = _trunc(top_str, 28)

        is_split = layer.is_auto_split
        if is_split:
            marker = " ==>"
        elif show_warnings:
            # Check if any top has fan-out > 1
            needs = False
            if fanout is not None:
                needs = any(top_fanout.get((i, j), (0, []))[0] > 1
                            for j in range(len(layer.tops)))
            else:
                needs = any(len(consumers.get(t, [])) > 1 for t in layer.tops)
            marker = " ***" if needs else "    "
        else:
            marker = "    "

        type_str = "Split [AUTO]" if is_split else layer.type

        print(f"{marker}{pos:>3} {layer.name:<32} {type_str:<14} "
              f"{bot_str:<30} {top_str:<30}")

        if show_warnings:
            for j, t in enumerate(layer.tops):
                if fanout is not None:
                    cnt, cons = top_fanout.get((i, j), (0, []))
                else:
                    cons = [c[0] for c in consumers.get(t, [])]
                    cnt = len(cons)
                if cnt > 1:
                    print(f"{'':>5}  ⚠ '{t}' fan-out={cnt} "
                          f"(producer='{fanout.layer_names.get(i, layer.name) if fanout else layer.name}'): "
                          f"{', '.join(cons)}")

        if is_split:
            src = layer.bottoms[0] if layer.bottoms else "?"
            print(f"{'':>5}  Split: {src} ──→ {', '.join(layer.tops)}")

        pos += 1

    print("-" * w)
    n_splits = sum(1 for l in net.layers if l.type == "Split")
    n_auto = sum(1 for l in net.layers if l.is_auto_split)
    n_other = n_splits - n_auto
    extra = f" (user-defined: {n_other})" if n_other else ""
    print(f"  Split layers: {n_splits} total, {n_auto} auto-inserted{extra}")
    print()


def print_fanout_analysis(net: NetSpec, mode: str = "caffe-ffi"):
    """Print a detailed fan-out analysis using accurate Pass 1 semantics
    (correctly tracks in-place last-producer updates)."""
    try:
        fa = analyze_fanout(net, mode)
    except ValueError as e:
        print(f"  ⚠ Fan-out analysis failed: {e}")
        return

    # Build reverse mapping: source_top -> list of consumer layer names
    source_to_consumers: dict[tuple[int, int], list[str]] = {}
    for (consumer_i, bottom_j), (src_i, src_j) in fa.bottom_to_source.items():
        source_to_consumers.setdefault((src_i, src_j), []).append(
            fa.layer_names.get(consumer_i, f"layer[{consumer_i}]")
        )

    w = 120
    print("-" * w)
    print("  Fan-out Analysis (before InsertSplits) — using Caffe last-producer semantics")
    print("-" * w)
    print(f"  {'Blob':<28} {'Producer':<25} {'#Con':>5}  {'Consumers':<50} {'Split?'}")
    print(f"  {'-'*28} {'-'*25} {'-'*5}  {'-'*50} {'-'*8}")

    n_needed = 0
    for (src_i, src_j, blob_name, cnt, needs) in fa.blob_records:
        if src_i == -1:
            prod = "<external input>"
        else:
            prod = fa.layer_names.get(src_i, f"?")
        cons = source_to_consumers.get((src_i, src_j), [])
        con_str = ", ".join(cons) if cons else "(unused, terminal)"
        con_str = con_str[:48] + "…" if len(con_str) > 49 else con_str
        needs_str = "YES ***" if needs else "no"
        if needs:
            n_needed += 1
        print(f"  {blob_name:<28} {prod:<25} {cnt:>5}  {con_str:<50} {needs_str}")

    print(f"\n  Total blobs needing split: {n_needed}")
    print()


def to_dot(net_before: NetSpec, net_after: NetSpec, mode: str = "caffe-ffi") -> str:
    """Generate Graphviz DOT representation showing before/after side by side."""
    lines = ['digraph InsertSplits {', '  rankdir=LR;', '  node [shape=box, style=filled];',
             '  compound=true;', '']

    def add_cluster(n, cluster_name, label, subgraph_id):
        lines.append(f'  subgraph cluster_{subgraph_id} {{')
        lines.append(f'    label="{label}";')
        lines.append(f'    style=dashed;')
        # Nodes
        for i, layer in enumerate(n.layers):
            color = "lightgreen" if layer.is_auto_split else "lightblue"
            shape = "ellipse" if layer.is_auto_split else "box"
            lines.append(f'    {subgraph_id}_{i} [label="{layer.name}\\n[{layer.type}]", '
                         f'fillcolor={color}, shape={shape}];')
        # External input node
        if mode == "caffe-ffi" and n.inputs:
            for idx, inp in enumerate(n.inputs):
                lines.append(f'    {subgraph_id}_in_{idx} [label="{inp}\\n[INPUT]", '
                             f'fillcolor=lightyellow, shape=oval];')
        # Edges
        node_id = {}
        for i, layer in enumerate(n.layers):
            node_id[layer.name] = f'{subgraph_id}_{i}'
        if mode == "caffe-ffi":
            for idx, inp in enumerate(n.inputs):
                node_id[inp] = f'{subgraph_id}_in_{idx}'

        # Build edges from blob connections
        # First, track which layer produces each blob
        blob_producer = {}
        if mode == "caffe-ffi":
            for idx, inp in enumerate(n.inputs):
                blob_producer[inp] = (f'{subgraph_id}_in_{idx}', inp)
        for i, layer in enumerate(n.layers):
            for t in layer.tops:
                blob_producer[t] = (f'{subgraph_id}_{i}', t)

        for i, layer in enumerate(n.layers):
            dst = f'{subgraph_id}_{i}'
            for b in layer.bottoms:
                if b in blob_producer:
                    src = blob_producer[b][0]
                    lines.append(f'    {src} -> {dst} [label="{b}"];')

        lines.append('  }')
        lines.append('')

    add_cluster(net_before, "before", "Before InsertSplits", "b")
    add_cluster(net_after, "after", "After InsertSplits", "a")
    lines.append('}')
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────
# Built-in test cases (extracted from test_insert_splits.cpp)
# ──────────────────────────────────────────────────────────────────────

BUILTIN_CASES: dict[str, str] = {}

def _case(name: str):
    def deco(fn):
        BUILTIN_CASES[name] = fn()
        return fn
    return deco


@_case("TwoConsumer")
def _():
    return """
name: "TwoConsumerNet"
input: "data"
input_shape { dim: 2 dim: 4 }
layer { name: "fc1" type: "InnerProduct" bottom: "data" top: "fc1_out"
  inner_product_param { num_output: 3 } }
layer { name: "fc2" type: "InnerProduct" bottom: "data" top: "fc2_out"
  inner_product_param { num_output: 3 } }
"""


@_case("InplaceTwoConsumer")
def _():
    return """
name: "InplaceNet"
input: "data"
input_shape { dim: 2 dim: 4 }
layer { name: "fc1" type: "InnerProduct" bottom: "data" top: "fc1_out"
  inner_product_param { num_output: 3 } }
layer { name: "relu1" type: "ReLU" bottom: "fc1_out" top: "fc1_out" }
layer { name: "fc2" type: "InnerProduct" bottom: "fc1_out" top: "fc2_out"
  inner_product_param { num_output: 3 } }
layer { name: "fc3" type: "InnerProduct" bottom: "fc1_out" top: "fc3_out"
  inner_product_param { num_output: 3 } }
"""


@_case("LinearChain")
def _():
    return """
name: "LinearNet"
input: "data"
input_shape { dim: 2 dim: 4 }
layer { name: "fc1" type: "InnerProduct" bottom: "data" top: "fc1_out"
  inner_product_param { num_output: 3 } }
layer { name: "relu" type: "ReLU" bottom: "fc1_out" top: "fc1_out" }
layer { name: "fc2" type: "InnerProduct" bottom: "fc1_out" top: "fc2_out"
  inner_product_param { num_output: 3 } }
"""


@_case("InputLayerThreeConsumer")
def _():
    return """
name: "InputLayerNet"
layer { name: "data" type: "Input" top: "data"
  input_param { shape { dim: 2 dim: 4 } } }
layer { name: "fc1" type: "InnerProduct" bottom: "data" top: "fc1_out"
  inner_product_param { num_output: 3 } }
layer { name: "fc2" type: "InnerProduct" bottom: "data" top: "fc2_out"
  inner_product_param { num_output: 3 } }
layer { name: "fc3" type: "InnerProduct" bottom: "data" top: "fc3_out"
  inner_product_param { num_output: 3 } }
"""


@_case("ExplicitSplit")
def _():
    return """
name: "ExplicitSplitNet"
input: "data"
input_shape { dim: 1 dim: 4 }
layer { name: "data_input_0_split" type: "Split" bottom: "data"
  top: "data_input_0_split_0" top: "data_input_0_split_1" }
layer { name: "fc1" type: "InnerProduct" bottom: "data_input_0_split_0" top: "out1"
  inner_product_param { num_output: 3 } }
layer { name: "fc2" type: "InnerProduct" bottom: "data_input_0_split_1" top: "out2"
  inner_product_param { num_output: 3 } }
"""


@_case("LossWeight")
def _():
    return """
name: "LossNet"
force_backward: true
input: "data"
input_shape { dim: 2 dim: 4 }
layer { name: "fc1" type: "InnerProduct" bottom: "data" top: "fc1_out"
  inner_product_param { num_output: 3 } loss_weight: 1.0 }
layer { name: "fc2" type: "InnerProduct" bottom: "fc1_out" top: "fc2_out"
  inner_product_param { num_output: 3 } }
layer { name: "fc3" type: "InnerProduct" bottom: "fc1_out" top: "fc3_out"
  inner_product_param { num_output: 3 } }
"""


@_case("DoubleInplace")
def _():
    return """
name: "DoubleInplaceNet"
input: "data"
input_shape { dim: 2 dim: 4 }
layer { name: "fc1" type: "InnerProduct" bottom: "data" top: "x"
  inner_product_param { num_output: 3 } }
layer { name: "relu1" type: "ReLU" bottom: "x" top: "x" }
layer { name: "relu2" type: "ReLU" bottom: "x" top: "x" }
layer { name: "fc2" type: "InnerProduct" bottom: "x" top: "fc2_out"
  inner_product_param { num_output: 3 } }
layer { name: "fc3" type: "InnerProduct" bottom: "x" top: "fc3_out"
  inner_product_param { num_output: 3 } }
"""


@_case("DataAndInplace")
def _():
    return """
name: "DataAndInplaceNet"
input: "data"
input_shape { dim: 2 dim: 4 }
layer { name: "fc1" type: "InnerProduct" bottom: "data" top: "fc1_out"
  inner_product_param { num_output: 3 } }
layer { name: "relu1" type: "ReLU" bottom: "fc1_out" top: "fc1_out" }
layer { name: "fc2" type: "InnerProduct" bottom: "fc1_out" top: "fc2_out"
  inner_product_param { num_output: 3 } }
layer { name: "fc3" type: "InnerProduct" bottom: "fc1_out" top: "fc3_out"
  inner_product_param { num_output: 3 } }
layer { name: "fc_branch" type: "InnerProduct" bottom: "data" top: "branch_out"
  inner_product_param { num_output: 3 } }
"""


@_case("Empty")
def _():
    return """
name: "EmptyNet"
input: "data"
input_shape { dim: 1 dim: 4 }
"""


@_case("BadRef")
def _():
    return """
name: "BadNet"
input: "data"
input_shape { dim: 1 dim: 4 }
layer { name: "fc1" type: "InnerProduct" bottom: "nonexistent" top: "out"
  inner_product_param { num_output: 3 } }
"""


# ──────────────────────────────────────────────────────────────────────
# Verification against expected split counts
# ──────────────────────────────────────────────────────────────────────

EXPECTED_SPLITS = {
    "TwoConsumer":           (1, ["data_input_0_split"]),
    "InplaceTwoConsumer":    (1, ["fc1_out_relu1_0_split"]),
    "LinearChain":           (0, []),
    "InputLayerThreeConsumer": (1, ["data_data_0_split"]),  # native Caffe: Input layer → splits after it
    "ExplicitSplit":         (0, []),  # no NEW splits (explicit split is user-provided)
    "LossWeight":            (1, ["fc1_out_fc1_0_split"]),
    "DoubleInplace":         (1, ["x_relu2_0_split"]),
    "DataAndInplace":        (2, ["data_input_0_split", "fc1_out_relu1_0_split"]),
    "Empty":                 (0, []),
    # BadRef: should raise error
}


def verify_case(name: str, proto_text: str, mode: str = "caffe-ffi") -> bool:
    """Run InsertSplits simulation and verify against expected split count.
    Returns True if verification passes.
    """
    try:
        net = parse_prototxt(proto_text)
        net_after = simulate_insert_splits(net, mode=mode)
    except ValueError as e:
        if name == "BadRef":
            print(f"  ✅ {name}: correctly raised error: {e}")
            return True
        print(f"  ❌ {name}: unexpected error: {e}")
        return False

    if name == "BadRef":
        print(f"  ❌ {name}: expected error but none raised")
        return False

    auto_splits = [l for l in net_after.layers if l.is_auto_split]
    expected_count, expected_names = EXPECTED_SPLITS.get(name, (None, None))

    ok = True
    if expected_count is not None:
        if len(auto_splits) != expected_count:
            print(f"  ❌ {name}: expected {expected_count} auto splits, got {len(auto_splits)}")
            ok = False
        else:
            actual_names = [l.name for l in auto_splits]
            missing = [n for n in expected_names if n not in actual_names]
            if missing:
                print(f"  ❌ {name}: missing expected splits: {missing}, got: {actual_names}")
                ok = False
            else:
                print(f"  ✅ {name}: {len(auto_splits)} auto split(s) {actual_names}")
    else:
        print(f"  ℹ️  {name}: {len(auto_splits)} auto split(s) (no expected count defined)")

    return ok


def get_split_info(net_after: NetSpec) -> dict:
    """Extract structured split information from a transformed network.

    Returns a JSON-serializable dict:
    {
        "total_layers": int,
        "auto_splits": [
            {"name": str, "source": str, "outputs": [str, ...]},
            ...
        ],
        "all_layer_names": [str, ...]
    }
    """
    auto_splits = []
    for l in net_after.layers:
        if l.is_auto_split:
            auto_splits.append({
                "name": l.name,
                "source": l.bottoms[0] if l.bottoms else "",
                "outputs": list(l.tops),
            })
    return {
        "total_layers": len(net_after.layers),
        "auto_split_count": len(auto_splits),
        "auto_splits": auto_splits,
        "all_layer_names": [l.name for l in net_after.layers],
    }


def verify_fixture_dir(fixture_dir: str, mode: str = "caffe-ffi",
                       verbose: bool = True) -> bool:
    """Validate all .prototxt fixtures in a directory against their .expected.json sidecars.

    Each .prototxt file can have a corresponding .expected.json (same basename)
    with the schema:
    {
        "auto_split_count": int,
        "auto_split_names": [str, ...],       # exact split layer names (optional, checked if present)
        "auto_split_sources": [str, ...],     # source blob names (optional)
        "expect_error": bool                   # if true, expect parse/simulate to raise error
    }
    """
    if not os.path.isdir(fixture_dir):
        print(f"❌ Fixture directory not found: {fixture_dir}")
        return False

    prototxt_files = sorted(glob.glob(os.path.join(fixture_dir, "*.prototxt")))
    if not prototxt_files:
        print(f"⚠ No .prototxt files found in {fixture_dir}")
        return True

    if verbose:
        print(f"Validating {len(prototxt_files)} fixture(s) from {fixture_dir}...\n")

    all_ok = True
    results = []

    for proto_path in prototxt_files:
        basename = os.path.basename(proto_path)
        name = os.path.splitext(basename)[0]
        expected_path = os.path.join(fixture_dir, f"{name}.expected.json")

        # Load expected JSON if exists
        expected = None
        if os.path.isfile(expected_path):
            with open(expected_path, encoding="utf-8") as f:
                expected = json.load(f)

        # Parse and simulate
        with open(proto_path, encoding="utf-8") as f:
            proto_text = f.read()

        try:
            net = parse_prototxt(proto_text)
            net_after = simulate_insert_splits(net, mode=mode)
            info = get_split_info(net_after)
            errored = False
            error_msg = None
        except (ValueError, Exception) as e:
            errored = True
            error_msg = str(e)
            info = {"auto_split_count": -1, "auto_splits": [], "all_layer_names": []}

        # Validate against expected
        ok = True
        issues = []

        if expected is None:
            status = "ℹ️  (no .expected.json — skipping validation)"
        elif expected.get("expect_error", False):
            if errored:
                status = f"✅ (expected error raised: {error_msg[:60]})"
            else:
                status = f"❌ expected error but simulation succeeded"
                ok = False
        else:
            if errored:
                status = f"❌ unexpected error: {error_msg}"
                ok = False
            else:
                exp_count = expected.get("auto_split_count")
                if exp_count is not None and info["auto_split_count"] != exp_count:
                    issues.append(f"split_count: expected {exp_count}, got {info['auto_split_count']}")
                    ok = False

                exp_names = expected.get("auto_split_names")
                if exp_names is not None:
                    actual_names = [s["name"] for s in info["auto_splits"]]
                    missing = [n for n in exp_names if n not in actual_names]
                    extra = [n for n in actual_names if n not in exp_names]
                    if missing or extra:
                        parts = []
                        if missing:
                            parts.append(f"missing: {missing}")
                        if extra:
                            parts.append(f"unexpected: {extra}")
                        issues.append(f"split_names: {'; '.join(parts)}")
                        ok = False

                exp_sources = expected.get("auto_split_sources")
                if exp_sources is not None:
                    actual_sources = [s["source"] for s in info["auto_splits"]]
                    missing = [s for s in exp_sources if s not in actual_sources]
                    if missing:
                        issues.append(f"split_sources missing: {missing}")
                        ok = False

                if ok:
                    splits_str = ", ".join(
                        f"{s['name']}({s['source']}→{len(s['outputs'])}outs)"
                        for s in info["auto_splits"]
                    ) or "none"
                    status = f"✅ {info['auto_split_count']} split(s): {splits_str}"
                else:
                    status = f"❌ {'; '.join(issues)}"

        if verbose:
            print(f"  {name:<30} {status}")

        results.append({"file": basename, "ok": ok, "info": info})
        all_ok = all_ok and ok

    if verbose:
        passed = sum(1 for r in results if r["ok"])
        total = len(results)
        print(f"\n{'='*60}")
        print(f"  Results: {passed}/{total} fixtures passed")
        if all_ok:
            print("  ✅ All fixtures PASSED")
        else:
            failed = [r["file"] for r in results if not r["ok"]]
            print(f"  ❌ Failed fixtures: {failed}")

    return all_ok


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def run_case(name: str, proto_text: str, mode: str = "caffe-ffi",
             dot: bool = False, verbose: bool = True):
    """Run a single case with full visualization."""
    if verbose:
        print(f"\n{'#'*100}")
        print(f"# Test Case: {name}")
        print(f"{'#'*100}\n")

    try:
        net = parse_prototxt(proto_text)
    except Exception as e:
        print(f"  Parse error: {e}")
        return None

    # Compute accurate fanout using Pass 1 semantics for before-view
    try:
        fanout_before = analyze_fanout(net, mode=mode)
    except ValueError:
        fanout_before = None

    if verbose:
        print_dag_table(net, "BEFORE InsertSplits", mode=mode, fanout=fanout_before)
        print_fanout_analysis(net, mode=mode)

    try:
        net_after = simulate_insert_splits(net, mode=mode)
    except ValueError as e:
        print(f"  ❌ InsertSplits error (expected for invalid protos): {e}\n")
        return None

    if verbose:
        print_dag_table(net_after, "AFTER InsertSplits", mode=mode, show_warnings=False)

        # Summary
        auto_splits = [l for l in net_after.layers if l.is_auto_split]
        if auto_splits:
            print(f"  📌 Auto-inserted Split layers:")
            for sl in auto_splits:
                print(f"     • {sl.name}: {sl.bottoms[0]} → {', '.join(sl.tops)}")
        else:
            print(f"  📌 No splits needed — network unchanged.")
        print()

    if dot and verbose:
        dot_str = to_dot(net, net_after, mode=mode)
        dot_file = f"dag_{name.lower()}.dot"
        with open(dot_file, "w", encoding="utf-8") as f:
            f.write(dot_str)
        print(f"  📄 Graphviz DOT saved to: {dot_file}")
        print(f"     Render with: dot -Tpng {dot_file} -o {dot_file.replace('.dot','.png')}")
        print()

    return net_after


def main():
    parser = argparse.ArgumentParser(
        description="InsertSplits DAG Visualizer — parse, simulate, and visualize Caffe split insertion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python viz_insert_splits.py                      # run + verify all built-in cases
          python viz_insert_splits.py --case TwoConsumer   # visualize specific case
          python viz_insert_splits.py mynet.prototxt       # visualize external prototxt
          python viz_insert_splits.py mynet.prototxt --dot # also generate Graphviz DOT
          python viz_insert_splits.py --mode native        # use native Caffe semantics
        """),
    )
    parser.add_argument("file", nargs="?", help="Path to a .prototxt file to visualize")
    parser.add_argument("--case", help="Run a specific built-in test case")
    parser.add_argument("--all", action="store_true", help="Visualize ALL built-in cases")
    parser.add_argument("--dot", action="store_true", help="Generate Graphviz DOT output")
    parser.add_argument("--mode", choices=["caffe-ffi", "native"], default="caffe-ffi",
                        help="InsertSplits mode (default: caffe-ffi)")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification on built-in cases (no full visualization)")
    parser.add_argument("--fixture-dir",
                        help="Validate all .prototxt fixtures in DIR against .expected.json sidecars")
    parser.add_argument("--emit-json", action="store_true",
                        help="Emit structured JSON results instead of text tables (for single file)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()

    # ── Fixture directory validation mode ──
    if args.fixture_dir:
        ok = verify_fixture_dir(args.fixture_dir, mode=args.mode, verbose=not args.quiet)
        if not ok:
            sys.exit(1)
        return

    # ── Verify mode ──
    if args.verify:
        print("Verifying InsertSplits simulation against expected results...\n")
        all_ok = True
        for name in BUILTIN_CASES:
            ok = verify_case(name, BUILTIN_CASES[name], mode=args.mode)
            all_ok = all_ok and ok
        print()
        if all_ok:
            print("✅ All cases PASSED")
        else:
            print("❌ Some cases FAILED")
            sys.exit(1)
        return

    # ── External file mode ──
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            proto_text = f.read()
        case_name = args.file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".prototxt", "")

        if args.emit_json:
            net = parse_prototxt(proto_text)
            net_after = simulate_insert_splits(net, mode=args.mode)
            info = get_split_info(net_after)
            info["name"] = case_name
            info["mode"] = args.mode
            print(json.dumps(info, indent=2))
            return

        run_case(case_name, proto_text, mode=args.mode, dot=args.dot)
        return

    # ── Specific built-in case ──
    if args.case:
        name = args.case
        if name not in BUILTIN_CASES:
            print(f"Unknown case '{name}'. Available: {', '.join(BUILTIN_CASES.keys())}")
            sys.exit(1)
        run_case(name, BUILTIN_CASES[name], mode=args.mode, dot=args.dot)
        return

    # ── All built-in cases ──
    if args.all:
        for name, proto_text in BUILTIN_CASES.items():
            run_case(name, proto_text, mode=args.mode, dot=args.dot)
        return

    # ── Default: verify all cases + show a couple visualizations ──
    print("InsertSplits DAG Visualizer")
    print("=" * 60)
    print()
    print("Running verification on all built-in test cases...\n")
    all_ok = True
    for name in BUILTIN_CASES:
        ok = verify_case(name, BUILTIN_CASES[name], mode=args.mode)
        all_ok = all_ok and ok
    print()
    if not all_ok:
        print("❌ Some cases FAILED — see details above.")
        sys.exit(1)
    print("✅ All cases PASSED.\n")

    # Show a few interesting visualizations
    print("=" * 100)
    print("Detailed visualization of key cases:")
    print("=" * 100)
    for showcase in ["TwoConsumer", "InplaceTwoConsumer", "DataAndInplace"]:
        run_case(showcase, BUILTIN_CASES[showcase], mode=args.mode, dot=args.dot)

    if not args.quiet:
        print("=" * 100)
        print("Tips:")
        print("  --case <name>    : Visualize a specific test case")
        print("  --all            : Visualize ALL built-in cases")
        print("  <file.prototxt>  : Visualize an external prototxt file")
        print("  --dot            : Also output Graphviz DOT files")
        print("  --verify         : Run verification only (no full DAG print)")
        print("=" * 100)


if __name__ == "__main__":
    main()
