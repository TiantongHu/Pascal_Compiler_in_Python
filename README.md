This project is a Pascal interpreter/compiler in Python. 

Please use `compiler_project.py --help` for more information on how to execute the program. 
Please check my other repo for a linux editor written in C, which provides PASCAL keyword highlighting 

It executes a subset of Pascal codes including arithmetic operation, writes, if statements, and procedures. It supports integer, real, boolean, and string.
Supports to inputs, functions, and loops are under development.

Like a real compiler, it consists of lexer, parser, and interpreter. It also simulates the Call Stack.
Of course python is not a language for compilers. This project is just for interest in compilers and a practice of Python, which is what I have been using for the last coop terms.

The main data structures used are stacks, tables, and Abstract Syntax Tree. A context-free grammar is developed for the purpose of understanding Pascal.

A set of Unittest cases are also provided. These cases are not complete, they were used during development for a few features.

