# SPDX-License-Identifier: GPL-2.0-only
"""bash 风格的 compose 变量插值。

迁移自上游 1.6.0 第 275-462 行 ``var_interpolate``，算法逐行翻译，未改变任何
token 化与解析语义。支持语法（对齐 docker-compose）：

* ``$VARIABLE`` / ``${VARIABLE}``
* ``${VARIABLE:-default}`` 未设置或为空时取默认值
* ``${VARIABLE-default}``  未设置时取默认值
* ``${VARIABLE:?err}``     未设置或为空时报错
* ``${VARIABLE?err}``      未设置时报错
* ``${VARIABLE:+alt}`` / ``${VARIABLE+alt}`` 替代展开
* ``$$`` 转义为字面量 ``$``；操作数支持任意嵌套

参考：
* https://docs.docker.com/compose/compose-file/#variable-substitution
* https://docs.docker.com/compose/env-file/
* https://www.gnu.org/software/bash/manual/html_node/Shell-Parameter-Expansion.html
"""

import string
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import PodmanComposeError

__all__ = ["var_interpolate"]


def var_interpolate(value: str, env: dict[str, Any]) -> str:
    var_name_chars = string.ascii_letters + string.digits + "_"
    var_name_start_chars = string.ascii_letters + "_"

    class VarInterpolationOperators(Enum):
        VAR_IF_NONEMPTY = ":-"
        VAR_IF_SET = "-"
        REQUIRED_SET = "?"
        REQUIRED_NONEMPTY = ":?"
        ALTERNATIVE1 = ":+"
        ALTERNATIVE2 = "+"

    operators = {op.value for op in VarInterpolationOperators}

    @dataclass
    class Token:
        def resolve(self, _: dict[str, str | None]) -> str:
            raise NotImplementedError()

    @dataclass
    class LiteralToken(Token):
        value: str

        def resolve(self, _: dict[str, str | None]) -> str:
            return self.value

    @dataclass
    class VarToken(Token):
        name: str
        operator: str | None
        operand: str | None

        def resolve(self, env: dict[str, str | None]) -> str:
            var_value = env.get(self.name)

            # This is only the case for simple $VAR or ${VAR} without any operator,
            # in which case we just return the variable value or empty string if not set
            if self.operator is None or self.operand is None:
                return var_value if var_value is not None else ""

            if self.operator == VarInterpolationOperators.REQUIRED_NONEMPTY.value:
                if var_value is None or var_value == "":
                    interpolated_operand = (
                        interpolate_str(self.operand, env) if self.operand else ""
                    )
                    raise PodmanComposeError(
                        f"required variable {self.name} is missing a value: {interpolated_operand}"
                    )
                return var_value

            if self.operator == VarInterpolationOperators.REQUIRED_SET.value:
                if var_value is None:
                    interpolated_operand = (
                        interpolate_str(self.operand, env) if self.operand else ""
                    )
                    raise PodmanComposeError(
                        f"required variable {self.name} is missing a value: {interpolated_operand}"
                    )
                return var_value

            if self.operator == VarInterpolationOperators.VAR_IF_NONEMPTY.value:
                condition = var_value is None or var_value == ""
                alternative = var_value if var_value is not None else ""
            elif self.operator == VarInterpolationOperators.VAR_IF_SET.value:
                condition = var_value is None
                alternative = var_value if var_value is not None else ""
            elif self.operator == VarInterpolationOperators.ALTERNATIVE1.value:
                condition = var_value is not None and var_value != ""
                alternative = ""
            elif self.operator == VarInterpolationOperators.ALTERNATIVE2.value:
                condition = var_value is not None
                alternative = ""
            else:
                raise ValueError(f"Unknown operator in variable interpolation: {self.operator}")

            return interpolate_str(self.operand, env) if condition else alternative

    def var_name_lookahead(start: int, chars: list[str]) -> tuple[int, str]:
        """
        moves the index to the end of the variable name
        returns variable name and position after variable name
        """
        var_name = ""
        i = start
        while i < len(chars) and chars[i] in var_name_chars:
            var_name += chars[i]
            i += 1
        return i, var_name

    def advance_to_closing_brace(start: int, chars: list[str]) -> int:
        i = start
        brace_level = 1
        while i < len(chars):
            char = chars[i]
            if char == "}":
                brace_level -= 1
                if brace_level == 0:
                    return i  # position of the closing brace
            elif char == "{":
                brace_level += 1
            i += 1
        raise ValueError("No closing brace found for variable interpolation")

    def resolve_brace_content(content: str) -> VarToken:
        operator = None
        operand = None

        # Check that the brace content starts with a valid variable name character.
        # Refuse to interpolate otherwise. This is how Docker behaves.
        if len(content) == 0 or content[0] not in var_name_start_chars:
            raise ValueError(
                f"Invalid interpolation format: ${{{content}}}."
                " You may need to escape any $ with another $"
            )

        i, name = var_name_lookahead(0, list(content))

        rest = content[i:]
        if rest:
            for op in operators:
                if rest.startswith(op):
                    operator = op
                    operand = rest[len(op) :]
                    break

            if operator is None:
                raise ValueError(f"Invalid variable interpolation syntax: ${{{content}}}")

        return VarToken(name=name, operator=operator, operand=operand)

    def tokenize(value: str) -> list[Token]:
        chars = list(value)
        tokens: list[Token] = []
        in_brace = False

        def append_text_char(char: str) -> None:
            if tokens and isinstance(tokens[-1], LiteralToken):
                tokens[-1].value += char
            else:
                tokens.append(LiteralToken(value=char))

        i = 0
        while i < len(chars):
            char = chars[i]
            if not in_brace:
                if char == "$":
                    # There is no lookahead, treat $ as literal
                    if i + 1 >= len(chars):
                        append_text_char(char)
                        i += 1
                        continue
                    lookahead_char = chars[i + 1]
                    if lookahead_char == "{":
                        in_brace = True
                        i += 2  # skip $ and {
                        continue
                    if lookahead_char == "$":
                        append_text_char("$")
                        i += 2  # skip both $$
                        continue
                    # If the lookahead char is valid for starting a variable name, parse the name.
                    # Otherwise, treat $ as literal (e.g. in "price is $5", $ should be literal)
                    if lookahead_char in var_name_start_chars:
                        i, var_name = var_name_lookahead(i + 1, chars)
                        tokens.append(VarToken(name=var_name, operator=None, operand=None))
                        continue  # already advanced i to a char after var name

                # Regular character
                append_text_char(char)
                i += 1
                continue

            # in_brace == True
            closing_index = advance_to_closing_brace(i, chars)
            brace_content = "".join(chars[i:closing_index])
            tokens.append(resolve_brace_content(brace_content))
            i = closing_index + 1  # move past the closing brace
            in_brace = False
            continue

        return tokens

    def interpolate_str(s: str, env: dict[str, str | None]) -> str:
        tokens = tokenize(s)
        resolved_parts = [token.resolve(env) for token in tokens]
        return "".join(resolved_parts)

    return interpolate_str(value, env)
