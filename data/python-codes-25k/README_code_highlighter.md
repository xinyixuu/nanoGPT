# code_highlighter.py README

This README explains how to use `code_highlighter.py` for 1:1 character-level
mapping of Python source code to various analyses. The script supports multiple
modes; each mode highlights a different facet of your code and preserves literal
newline characters.

---

## Prerequisites

- Python 3.8+ (uses `tokenize`, `ast` from standard library)

---

## Installation

1. Place `code_highlighter.py` in your working directory.
2. (Optional) Make it executable:
```bash
chmod +x code_highlighter.py
```

---

## Usage

```bash
python code_highlighter.py [--mode <mode>] <path/to/source.py>
```

- `--mode`: one of the supported modes (default: `general`).
- `<path/to/source.py>`: the Python file to analyze.

On completion, the script writes:
- **`<source>.mapped`**: the mapped output file (one character per original character).
- **Console**: confirms length and prints a legend mapping each marker back to its category.

---

## Examples:
### general

Categories:
```
# Legend: char → category
A = WHITESPACE
B = COMMENT
C = KEYWORD
D = NAME
E = OP
F = INDENT
G = DEDENT
H = ENDMARKER
```

Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```

```
BBBBBBBBBBBBBBB
CCCCCADE
FFFFCCCADEDDDDEADEE
FFFFFFFFCCCCCCADDDDEDDDDDEDDDDDEDE
```

### exact

Categories:
```
# Legend: char → category
A = WHITESPACE
B = COMMENT
C = NAME
D = COLON
E = INDENT
F = LPAR
G = COMMA
H = RPAR
I = DOT
J = DEDENT
K = ENDMARKER
```

Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```
```
BBBBBBBBBBBBBBB
CCCCCACD
EEEECCCACFCCCCGACHD
EEEEEEEECCCCCCACCCCICCCCCICCCCCFCH
```
### keywords

Categories
```
# Legend: char → category
A = WHITESPACE
B = COMMENT
C = KW_CLASS
D = NAME
E = COLON
F = INDENT
G = KW_DEF
H = LPAR
I = COMMA
J = RPAR
K = KW_RETURN
L = DOT
M = DEDENT
N = ENDMARKER
```

Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```
```
BBBBBBBBBBBBBBB
CCCCCADE
FFFFGGGADHDDDDIADJE
FFFFFFFFKKKKKKADDDDLDDDDDLDDDDDHDJ
```

### nesting

Categories:
```
# Legend: char → category
A = LEVEL_0
B = LEVEL_1
C = LEVEL_2
# etc...
```

Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```
```
AAAAAAAAAAAAAAA
AAAAAAAA
BBBBBBBBBBBBBBBBBBB
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
```

### param_nesting
Categories:
```
# Legend: char → category
A = LEVEL_1
```

Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```
```
         AAAAAAAAA
                               AAA
```

### argnum
Categories:
```
# Legend: char → category
A = ARG_1
B = ARG_2
# etc...
```
Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```
```
          AAAA  B
                                A
```

### dot_nesting

Categories:
```
# Legend: char → category
A = DOT
B = DEPTH_1
C = DEPTH_2
D = DEPTH_3
# etc...
```
Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```
```

               BBBBACCCCCADDDDD
```

### name_kind

Categories:
```
# Legend: char → category
A = WHITESPACE
B = CLASS
C = FUNC
D = VAR
```

Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```
```
AAAAAAAAAAAAAAA
AAAAAABA
AAAAAAAACADDDDAADAA
AAAAAAAAAAAAAAADDDDADDDDDADDDDDADA
```

### literals

Categories:
```
# Legend: char → category
A = WHITESPACE
B = FLOAT
C = INT
```

Example:
```
x = 1.0 + 2
y = 3 x 5
```
```
AAAABBBAAAC
AAAACAAAC
```

### semantic

Categories:
```
# Legend: char → category
A = WHITESPACE
B = COMMENT
C = DECL
D = NAME
E = PUNCT
F = WS
G = CONTROL
```

Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```
```
BBBBBBBBBBBBBBB
CCCCCADE
FFFFCCCADEDDDDEADEE
FFFFFFFFGGGGGGADDDDEDDDDDEDDDDDEDE
```

### comments

Categories:
```
# Legend: char → category
A = WHITESPACE
B = COMMENT
C = CODE
```

Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```
```
BBBBBBBBBBBBBBB
CCCCCACC
CCCCCCCACCCCCCCACCC
CCCCCCCCCCCCCCACCCCCCCCCCCCCCCCCCC
```

### scope

Categories:
```
# Legend: char → category
A = WHITESPACE
B = GLOBAL
C = CLASS
```

Example:
```py
# Example Class
class C:
    def m(self, x):
        return self.attr1.attr2(x)
```
```
BBBBBBBBBBBBBBB
CCCCCACC
CCCCCCCACCCCCCCACCC
CCCCCCCCCCCCCCACCCCCCCCCCCCCCCCCCC
```
