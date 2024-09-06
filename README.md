# Regular Expression Engine

This project implements a regular expression engine that can convert regular expressions to NFAs and DFAs, and simulate string matching against the resulting automata.

# Author

Rodolfo Lopez

# Date

November 12, 2021

## Features

- Convert regular expressions to abstract syntax trees
- Convert syntax trees to NFAs
- Convert NFAs to DFAs
- Simulate string matching against DFAs

## Usage

The main class to use is `RegEx`:

```python
from pa3 import RegEx
```

Create RegEx object from file containing regex

```python
regex = RegEx("regex1.txt")
```

Simulate string matching

```python
result = regex.simulate("test string")
```

The `RegEx` constructor takes a filename containing the regular expression specification. The file should have two lines:

1. The alphabet symbols
2. The regular expression

The `simulate` method takes an input string and returns True if it matches the regex, False otherwise.

## Implementation Details

The conversion process follows these steps:

1. Parse regex to abstract syntax tree
2. Convert syntax tree to NFA
3. Convert NFA to DFA
4. Simulate string matching on DFA

Key classes:

- `RegEx`: Main interface
- `BinTree`: Binary tree for syntax tree
- `NFA`: NFA representation and conversion to DFA
- `DFA`: DFA representation and simulation

## Testing

The `test_pa3.py` script runs tests on sample regex and string inputs. To run:

```bash
python test_pa3.py
```

## Limitations

- Does not support all advanced regex features like backreferences
- Performance may degrade for very complex expressions

## Acknowledgments

Professor John Glick wrote all test scripts.
