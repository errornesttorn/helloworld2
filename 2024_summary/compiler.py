# -----------------------------------------------------------------------------
# Code consist of three chapters. First is mostly ChatGPT generated parcing
# and preprocessing functions, Chatpter 2 consist of main compiler functions -
# evaluate_expression responsible for compiling expressions and compile
# responsible for compiler the control flow and non expressive statements.
# Third chapter are examples
# -----------------------------------------------------------------------------


import re
import keyword


# -----------------------------------------------------------------------------
# Chapter 1 - ChatGPT generated parsing and preproccessing functions
# https://chatgpt.com/share/3a88e125-d91d-4b06-9110-66d0278c20f9
# -----------------------------------------------------------------------------

def truncate_to_expressions(c_code):
    # Remove single-line comments (// ...)
    c_code = re.sub(r'//.*', '', c_code)
    
    # Remove multi-line comments (/* ... */)
    c_code = re.sub(r'/\*[\s\S]*?\*/', '', c_code)
    
    token_pattern = re.compile(r"""
        (?P<keyword>\b(if|else|for|while|return|do)\b) |
        (?P<type>\b(int|float|double|char|void|long|short|unsigned|signed)\b) |
        (?P<identifier>[a-zA-Z_][a-zA-Z0-9_]*) |
        (?P<number>\b\d+(\.\d+)?\b) |
        (?P<operator>[+\-*/%=&|!<>^]+) |
        (?P<punctuation>[;,.(){}\[\]]) |
        (?P<whitespace>\s+) |
        (?P<string>\".*?\")
    """, re.VERBOSE)

    expressions = []
    current_expression = ""

    force_enters_pattern = re.compile(r'\b(if|else|for|while|return|do)\b\s*\(\s*.*?\s*\)\s*$')
    for_bar = 0

    for match in token_pattern.finditer(c_code):
        token_type = match.lastgroup
        token_value = match.group(token_type)
        
        if token_type == "whitespace":
            if force_enters_pattern.match(current_expression):
                current_expression += token_value
                expressions.append(current_expression.strip())
                current_expression = ""
            continue

        # if token_value == "for" and for_bar == 0:
        #     for_bar = 1
        if for_bar and token_value == "(":
            for_bar += 1
        if for_bar and token_value == ")":
            for_bar -= 1
            if for_bar == 1:
                current_expression += token_value
                expressions.append(current_expression.strip())
                current_expression = ""
                continue
        

        if token_value == "{":
            if current_expression.strip():
                expressions.append(current_expression.strip())
                current_expression = ""
            expressions.append("{")
        elif token_value == "else":
            if current_expression.strip():
                expressions.append(current_expression.strip())
                current_expression = ""
            expressions.append("else")
        elif token_value == ";":
            current_expression += token_value
            expressions.append(current_expression.strip())
            current_expression = ""
        elif token_value == "}":
            if current_expression.strip():
                expressions.append(current_expression.strip())
                current_expression = ""
            expressions.append("}")
        else:
            current_expression += token_value + " "
    
    if current_expression.strip():
        expressions.append(current_expression.strip())
    
    return expressions


def enforce_braces(expressions):
    modified_expressions = []
    control_statements = {"if", "else", "for", "while", "do"}

    i = 0
    while i < len(expressions):
        expr = expressions[i].strip()
        
        # Check if the expression is a control statement
        if any(expr.startswith(keyword) for keyword in control_statements):
            # Check if the next expression is not an opening brace
            if i + 1 < len(expressions) and expressions[i + 1] != "{":
                modified_expressions.append(expr)
                modified_expressions.append("{")
                
                # Find where the statement ends (either next semicolon or brace)
                j = i + 1
                while j < len(expressions) and not expressions[j].startswith("{") and not expressions[j].startswith("}"):
                    j += 1

                modified_expressions.append(expressions[i + 1].strip())
                modified_expressions.append("}")
                
                # Skip to the next expression after the processed statement
                i = j
            else:
                modified_expressions.append(expr)
        else:
            modified_expressions.append(expr)
        
        i += 1
    
    return modified_expressions


def simplify_declarations(expressions):
    simplified_expressions = []
    
    # Pattern to match int declarations with assignments
    declaration_pattern = re.compile(r'\b(int|float|double|char|void|long|short|unsigned|signed)\b\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*);')
    
    for expression in expressions:
        match = declaration_pattern.match(expression)
        if match:
            var_type = match.group(1)
            var_name = match.group(2)
            var_value = match.group(3)
            
            # Add the separated lines to the result list
            simplified_expressions.append(f"{var_type} {var_name};")
            simplified_expressions.append(f"{var_name} = {var_value};")
        else:
            # If it doesn't match, keep the expression as is
            simplified_expressions.append(expression)
    
    return simplified_expressions


def simplify_for_loop_declarations(expressions):
    simplified_expressions = []
    
    # Pattern to match for loop declarations with assignments
    for_declaration_pattern = re.compile(r'\bfor\s*\(\s*(int|float|double|char|void|long|short|unsigned|signed)\b\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)')
    
    for expression in expressions:
        match = for_declaration_pattern.match(expression)
        if match:
            var_type = match.group(1)
            var_name = match.group(2)
            var_value = match.group(3)
            
            # Add the separated lines to the result list
            simplified_expressions.append(f"{var_type} {var_name};")
            simplified_expressions.append(f"for ({var_name} = {var_value}")
        else:
            # If it doesn't match, keep the expression as is
            simplified_expressions.append(expression)
    
    return simplified_expressions


def replace_true_false(expressions):
    replaced_expressions = []
    
    # Patterns to match true and false keywords
    true_pattern = re.compile(r'\btrue\b')
    false_pattern = re.compile(r'\bfalse\b')
    
    for expression in expressions:
        # Replace true with 1
        expression = true_pattern.sub('1', expression)
        # Replace false with 0
        expression = false_pattern.sub('0', expression)
        
        # Add the modified expression to the list
        replaced_expressions.append(expression)
    
    return replaced_expressions


def replace_increment_decrement(expressions):
    modified_expressions = []
    
    # Patterns for matching post-increment (variable++) and pre-increment (++variable)
    post_increment_pattern = re.compile(r'(\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*\+\+')
    pre_increment_pattern = re.compile(r'\+\+\s*(\b[a-zA-Z_][a-zA-Z0-9_]*\b)')
    
    # Patterns for matching post-decrement (variable--) and pre-decrement (--variable)
    post_decrement_pattern = re.compile(r'(\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*--')
    pre_decrement_pattern = re.compile(r'--\s*(\b[a-zA-Z_][a-zA-Z0-9_]*\b)')

    for expression in expressions:
        # Replace post-increment with cmp_varpp(variable)
        expression = post_increment_pattern.sub(r'cmp_varpp(\1)', expression)
        
        # Replace pre-increment with cmp_ppvar(variable)
        expression = pre_increment_pattern.sub(r'cmp_ppvar(\1)', expression)
        
        # Replace post-decrement with cmp_varmm(variable)
        expression = post_decrement_pattern.sub(r'cmp_varmm(\1)', expression)
        
        # Replace pre-decrement with cmp_mmvar(variable)
        expression = pre_decrement_pattern.sub(r'cmp_mmvar(\1)', expression)
        
        modified_expressions.append(expression)
    
    return modified_expressions


def extract_functions(c_code):
    functions = []
    current_function = []
    inside_function = False
    brace_count = 0

    # Split the code into characters to track braces properly
    i = 0
    while i < len(c_code):
        char = c_code[i]

        if char == '{' and not inside_function:
            inside_function = True
            brace_count = 1  # Starting a function
            current_function.append(char)
        
        elif inside_function:
            current_function.append(char)

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            if brace_count == 0:
                # End of the function
                functions.append("".join(current_function))
                current_function = []
                inside_function = False

        else:
            # If not inside a function, check if a new function is starting
            if char.isalpha():
                # Check for the start of a function by finding '(' followed by '{'
                start_index = i
                while i < len(c_code) and c_code[i] != '{':
                    i += 1
                if i < len(c_code) and c_code[i] == '{':
                    current_function.append(c_code[start_index:i])
                    current_function.append('{')
                    brace_count = 1
                    inside_function = True

        i += 1

    return functions


def parse_function(function_str):
    # Regex pattern to capture the function signature
    signature_pattern = r'(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{'
    match = re.match(signature_pattern, function_str)

    if not match:
        raise ValueError("The provided string does not match a valid C function signature.")

    # Extract function name and parameters from the signature
    return_type, func_name, params_str = match.groups()

    # Split the parameters into individual variables
    params = [param.split()[-1].strip('*') for param in params_str.split(',') if param]

    # Extract the function body
    body_start = function_str.find('{') + 1
    body_end = function_str.rfind('}')
    function_body = function_str[body_start:body_end]

    return func_name, params, function_body


def parse_expression_inner(expr):
    expr = expr.strip()

    # Simplify top-level parentheses
    expr = simplify_parentheses(expr)

    # Operators sorted by precedence
    precedence = [
        ['='],                # Assignment
        ['+', '-'],            # Addition and Subtraction
        ['*', '/', '%'],       # Multiplication, Division, Modulus
        ['<<', '>>'],          # Bitwise shifts
        ['&'],                 # Bitwise AND
        ['^'],                 # Bitwise XOR
        ['|'],                 # Bitwise OR
        ['&&'],                # Logical AND
        ['||'],                # Logical OR
        ['==', '!=', '<=', '>=', '<', '>'],  # Comparison
    ]

    # Improved precedence
    precedence = [
        ['='],                # Assignment
        ['||', '$'],              # Logical OR
        ['&&', '#'],              # Logical AND
        ['|'],                    # Bitwise OR
        ['^'],                    # Bitwise XOR
        ['&'],                    # Bitwise AND
        ['==', '!=', '@', ":"],  # Equality
        ['<', '<=', '>', '>=', '`', '?'],   # Relational operators
        ['<<', '>>'],             # Bitwise shifts
        ['+', '-'],               # Addition and Subtraction
        ['*', '/', '%'],          # Multiplication, Division, Modulus
    ]
    
    # Find the operator with the lowest precedence at the top level
    for ops in precedence:
        for op in ops:
            sub_expressions = split_top_level(expr, op)
            if len(sub_expressions) > 1:
                return op, sub_expressions
    
    # Handle dereferencing (*) and address-of (&) as special unary operators
    if expr.startswith('*') or expr.startswith('&'):
        return handle_unary_operators(expr)

    # Handle indexing operation
    if is_indexing_operation(expr):
        variable_name, index_expr = extract_indexing_operation(expr)
        return 'index', [variable_name, index_expr]

    # If not a binary operation, check for a function call
    if is_function_call(expr):
        function_name, args = extract_function_call(expr)
        sub_expressions = split_arguments(args)
        return function_name, sub_expressions

    return None, [expr]

def handle_unary_operators(expr):
    if expr.startswith('*'):
        sub_expr = expr[1:].strip()
        return 'dereference', [sub_expr]
    elif expr.startswith('&'):
        sub_expr = expr[1:].strip()
        return 'addressof', [sub_expr]
    return None, [expr]

def simplify_parentheses(expr):
    while expr.startswith('(') and expr.endswith(')'):
        # Remove the top-level parentheses and check if it's still valid
        balance = 0
        for i, char in enumerate(expr):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            if balance == 0 and i < len(expr) - 1:
                # Found a closing parenthesis before the end, meaning it's not a top-level parenthesis
                return expr
        # If we completed the loop, the entire expression was wrapped in parentheses
        expr = expr[1:-1].strip()
    return expr

def split_arguments(args):
    sub_expressions = []
    current = ''
    balance = 0
    for char in args:
        if char == ',' and balance == 0:
            sub_expressions.append(current.strip())
            current = ''
        else:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            current += char
    if current:
        sub_expressions.append(current.strip())
    return sub_expressions

def split_top_level(expr, op):
    sub_expressions = []
    current = ''
    balance = 0
    op_len = len(op)
    i = 0
    while i < len(expr):
        if expr[i:i+op_len] == op and balance == 0:
            # Ensure the operator is used in a binary context (not unary)
            if (i == 0 or expr[i-1] not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_)]') and op in ['*', '&']:
                # Ignore this as it's likely a unary operator
                current += expr[i:i+op_len]
            else:
                sub_expressions.append(current.strip())
                current = ''
            i += op_len - 1
        else:
            if expr[i] == '(' or expr[i] == '[':
                balance += 1
            elif expr[i] == ')' or expr[i] == ']':
                balance -= 1
            current += expr[i]
        i += 1
    if current:
        sub_expressions.append(current.strip())
    return sub_expressions

def is_function_call(expr):
    """Check if an expression is a function call."""
    match = re.match(r'^([a-zA-Z_]\w*)\s*\(', expr)
    if match:
        balance = 0
        for i, char in enumerate(expr):
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
                if balance == 0:
                    # Check if the closing parenthesis is at the end of the function call
                    if i == len(expr) - 1:
                        return True
                    else:
                        return False
    return False

def extract_function_call(expr):
    """Extract the function name and arguments from a function call expression."""
    match = re.match(r'^([a-zA-Z_]\w*)\s*\((.*)\)$', expr)
    if match:
        function_name = match.group(1)
        args = match.group(2)
        return function_name, args
    return None, None

def is_indexing_operation(expr):
    """Check if an expression is an indexing operation."""
    balance = 0
    for i in range(len(expr)):
        if expr[i] == '[':
            balance += 1
            if balance == 1:  # First opening bracket found
                # Check if there is a corresponding closing bracket at the end
                if expr[-1] == ']' and balance == 1:
                    return True
                break
        elif expr[i] == ']':
            balance -= 1
    return False

def extract_indexing_operation(expr):
    """Extract the variable name and index expression from an indexing operation."""
    balance = 0
    for i in range(len(expr)):
        if expr[i] == '[':
            balance += 1
            if balance == 1:  # First opening bracket found
                variable_name = expr[:i].strip()
                index_expr = expr[i+1:-1].strip()  # Strip out the index expression
                return variable_name, index_expr
        elif expr[i] == ']':
            balance -= 1
    return None, None


def parse_expression(expr):
    expr = expr.replace(" ", "")
    expr = expr.replace('==', '@')
    expr = expr.replace('!=', ':')
    expr = expr.replace('<=', '`')
    expr = expr.replace('>=', '?')
    expr = expr.replace('&&', '#')
    expr = expr.replace('||', '$')
    operator, subexpressions = parse_expression_inner(expr)
    new_subexpressions = []
    for subexpression in subexpressions:
        rebuild = subexpression
        rebuild = rebuild.replace('@', '==')
        rebuild = rebuild.replace(':', '!=')
        rebuild = rebuild.replace('`', '<=')
        rebuild = rebuild.replace('?', '>=')
        rebuild = rebuild.replace('#', '&&')
        rebuild = rebuild.replace('$', '||')
        new_subexpressions.append(rebuild)
    if operator in ['@', ":", '`', '?', '#', '$']:
        operator = {
            '@': '==',
            ':': '!=',
            '`': '<=',
            '?': '>=',
            '#': '&&',
            '$': '||',
        }.get(operator)
    if operator == '=' and (new_subexpressions[0][-1] in ['+', '-', '*', '/', '%', '|', '&']):
        new_subexpressions = [new_subexpressions[0][:-1], new_subexpressions[0]+new_subexpressions[1]]
    
    if len(new_subexpressions) > 2:
        if operator == '=':
            # Right-to-left associativity: join from right to left
            new_expression = f"{operator.join(new_subexpressions[1:])}"
            return operator, [new_subexpressions[0], new_expression]
        else:
            # Left-to-right associativity: join from left to right
            new_expression = f"{operator.join(new_subexpressions[:-1])}"
            return operator, [new_expression, new_subexpressions[-1]]

    return operator, new_subexpressions


# -----------------------------------------------------------------------------
# Chapter 2 - Actual compiler logic
# -----------------------------------------------------------------------------

RET = 0
FRAME = 1
STACK = 2
TEMP = 3


def instruction(instruction: str, operands=None, comment=None):
    if operands is None:
        operands = []
    number_of_operands = len(operands)
    if instruction in ['movaf', 'movad', 'pusha', 'popb']:
        assert number_of_operands == 0, f'Za dużo operandów dla instrukcji {instruction}. Powinno być 0, otrzymano {number_of_operands}'
    if instruction in ['addac']:
        assert number_of_operands == 1, f'Zła liczba operandów dla instrukcji {instruction}. Powinno być 1, otrzymano {number_of_operands}'
    return [{
        'type': 'instruction',
        'instruction': instruction,
        'operands': operands,
        'comment': comment,
    }]


# classless constructors. I hate classes
def comment(comment: str):
    return [{
        'type': 'comment',
        'comment': comment,
    }]

def label(label: str):
    return [{
        'type': 'label',
        'label': label,
    }]


# chatgpt generated
def classify_c_string(input_str):
    # Define regex for a valid C number (integer, floating-point, or hexadecimal)
    c_number_regex = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$|^0[xX][0-9a-fA-F]+$'
    # Define regex for a valid C variable name
    c_variable_name_regex = r'^[a-zA-Z_]\w*$'
    # Check if input is a valid C number
    if re.match(c_number_regex, input_str):
        return 'number'
    # Check if input is a valid C variable name and not a C keyword
    if re.match(c_variable_name_regex, input_str) and not keyword.iskeyword(input_str):
        return 'name'
    # If none of the above, return 'other'
    return 'other'


# registers: a, b, f, s
# dereferencing: a->d/u, b->?/j
# operanded: c(k), r, e(g)


def decode_name(variable_name, name_dict, mov_mode=False):
    variable_object = name_dict.get(variable_name)
    if isinstance(variable_object, int) and mov_mode:
        return 'g', name_dict.get(variable_name)
    if isinstance(variable_object, int):
        return 'e', name_dict.get(variable_name)
    assert variable_object.get('type'), 'variable_object must have a type parameter'
    assert variable_object.get('address') is not None, 'variable_object holds no address'
    address = variable_object.get('address')
    if variable_object.get('type') == 'local' and mov_mode:
        return 'g', address
    if variable_object.get('type') == 'global' and mov_mode:
        return 'k', address
    if variable_object.get('type') == 'local':
        return 'e', address
    if variable_object.get('type') == 'global':
        return 'r', address
    else:
        raise Exception('unknown variable_object type')


def evaluate_expression(expression, name_dict, machine, jump_counter_object, is_top_level=False, jumpy_task=False, reverse_jumpy_task=False):
    print('Ekspresja:', expression)
    if jump_counter_object is None:
        raise ValueError('jump_counter_object is required')
    operator, subexpressions = parse_expression(expression)
    if operator == 'index':
        operator, subexpressions = parse_expression('*(' + "+".join(subexpressions) + ')')
    number_of_subexpressions = len(subexpressions)
    com = f" {operator} ".join([sub for sub in subexpressions])
    generated_code = comment(f'Evaluating {com}')
    generated_code = []

    
    if operator is None:
        assert number_of_subexpressions == 1, 'more than 1 subexpression despite operator being None'
        subexpression = subexpressions[0]
        if classify_c_string(subexpression) == 'number':
            generated_code += instruction('movac', operands=[subexpression])
            return generated_code
        elif classify_c_string(subexpression) == 'name' and machine.get('eac'):
            variable_name = subexpression
            pronoun, address = decode_name(variable_name, name_dict)
            generated_code += instruction(f'mova{pronoun}', operands=[address])
            return generated_code
        elif classify_c_string(subexpression) == 'name':
            variable_name = subexpression
            pronoun, address = decode_name(variable_name, name_dict)
            # generated_code += comment(f'Obliczamy wartość {variable_name}')
            if pronoun == 'e':
                generated_code += instruction('movaf', comment='przenosimy wartość frame do ret')
                generated_code += instruction('addac', operands=[address], comment=f"przesuwamy wartość frame w ret o odległość '{variable_name}' żeby uzyskać wskaźnik na {variable_name}")
                generated_code += instruction('movad', comment=f"bierzemy wartość '{variable_name}' do ret")
                return generated_code
            else:
                generated_code += instruction('movar', operands=[address])
                return generated_code
        else:
            raise Exception(f'Otrzymano ekspresję bez operatora. Oto ekspresja: {expression}')

    
    if operator == "=":
        assert number_of_subexpressions == 2, f'"=" operator got a wrong number of subexpressions. Exprected 2, got {number_of_subexpressions}'
        # evaluate the left side first to know if theres anything to push onto the stack
        # check if the lvalue is a dereference
        left_operator, left_subexpressions = parse_expression(subexpressions[0])
        if left_operator == 'index':
            left_operator, left_subexpressions = parse_expression('*(' + "+".join(left_subexpressions) + ')')
        assert len(left_subexpressions) == 1, "Only dereferences or variables can be assigned"
        assert left_operator is None or left_operator == "dereference", "Only dereferences or variables can be assigned"
        
        # left operand
        left_type = 'expression'
        if left_operator == 'dereference' and classify_c_string(left_subexpressions[0]) == 'number':
            subexpressions[0] = left_subexpressions[0]
            left_type = 'number'
        elif left_operator == 'dereference':
            subexpressions[0] = left_subexpressions[0]
        elif classify_c_string(subexpressions[0]) == 'name' and machine.get('eac'):
            pronoun, address = decode_name(subexpressions[0], name_dict, mov_mode=True)
            left_type = 'name'
        elif classify_c_string(subexpressions[0]) == 'name':
            subexpressions[0] = "addressof(" + subexpressions[0] + ")"
        
        # top level optimalization
        if is_top_level and classify_c_string(subexpressions[1]) == 'number' and left_type == 'number':
            generated_code += instruction('movkc', operands=[subexpressions[0], subexpressions[1]])
            return generated_code
        elif is_top_level and machine.get('eac'):
            if classify_c_string(subexpressions[1]) == 'name' and left_type == 'number':
                pronoun, address = decode_name(subexpressions[1], name_dict)
                generated_code += instruction(f'movk{pronoun}', operands=[subexpressions[0], address])
                return generated_code
            elif classify_c_string(subexpressions[1]) == 'name' and left_type == 'name':
                pronoun_l, address_l = decode_name(subexpressions[0], name_dict, mov_mode=True)
                pronoun_r, address_r = decode_name(subexpressions[1], name_dict)
                generated_code += instruction(f'mov{pronoun_l}{pronoun_r}', operands=[address_l, address_r])
                return generated_code
            elif classify_c_string(subexpressions[1]) == 'number' and left_type == 'name':
                pronoun, address = decode_name(subexpressions[0], name_dict, mov_mode=True)
                generated_code += instruction(f'mov{pronoun}c', operands=[address, subexpressions[1]])
                return generated_code
            elif classify_c_string(subexpressions[1]) == 'name' and left_type == 'expression':
                pronoun, address = decode_name(subexpressions[1], name_dict)
                generated_code += evaluate_expression(subexpressions[0], name_dict, machine, jump_counter_object)
                generated_code += instruction(f'movu{pronoun}', operands=[address])
                return generated_code
            elif classify_c_string(subexpressions[1]) == 'number' and left_type == 'expression':
                generated_code += evaluate_expression(subexpressions[0], name_dict, machine, jump_counter_object)
                generated_code += instruction('movuc', operands=[subexpressions[1]])
                return generated_code

        # evaluating and pushing if an expression
        if left_type == 'expression':
            generated_code += evaluate_expression(subexpressions[0], name_dict, machine, jump_counter_object)
            generated_code += instruction('pusha')

        # right operand
        generated_code += evaluate_expression(subexpressions[1], name_dict, machine, jump_counter_object) # evaluating no matter what

        # final assignment
        if left_type == 'expression':
            generated_code += instruction('popb')
            generated_code += instruction('movja')
        elif left_type == 'number':
            generated_code += instruction('movka', operands=[subexpressions[0]])
        elif left_type == 'name':
            generated_code += instruction(f'mov{pronoun}a', operands=[address])
        return generated_code

        
    if operator in ['+', '-', '==', '!=', '<=', '>=', '<', '>']\
        or (operator=='*' and machine.get('has_mul'))\
        or (operator=='/' and machine.get('has_div'))\
        or (operator=='%' and machine.get('has_mod'))\
        or (operator=='<<' and machine.get('has_shl'))\
        or (operator=='>>' and machine.get('has_shr'))\
        or (operator=='&' and machine.get('has_and'))\
        or (operator=='^' and machine.get('has_xor'))\
        or (operator=='|' and machine.get('has_or')):
        assert number_of_subexpressions == 2, f'"{operator}" operator has a wrong number of subexpressions. Exprected 2, got {number_of_subexpressions}'
        left_type = classify_c_string(subexpressions[0])
        right_type = classify_c_string(subexpressions[1])
        if (left_type == 'name' and not machine.get('eac') or (left_type == 'name' and operator == '=')):
            left_type = 'other'
        if right_type == 'name' and not machine.get('eac'):
            right_type = 'other'
        if left_type == 'other' and right_type == 'other':
            generated_code += evaluate_expression(subexpressions[1], name_dict, machine, jump_counter_object)
            generated_code += instruction('pusha')
            generated_code += evaluate_expression(subexpressions[0], name_dict, machine, jump_counter_object)
            generated_code += instruction('popb')
            pronouns = 'ab'
            operands = []
        elif left_type == 'other':
            generated_code += evaluate_expression(subexpressions[0], name_dict, machine, jump_counter_object)
            if right_type == 'name':
                pronoun, address = decode_name(subexpressions[1], name_dict)
                pronouns = f'a{pronoun}'
                operands = [address]
            elif right_type == 'number':
                number = subexpressions[1]
                pronouns = 'ac'
                operands = [number]
            else:
                raise Exception("This should never execute. Something is wrong")
        elif right_type == 'other':
            generated_code += evaluate_expression(subexpressions[1], name_dict, machine, jump_counter_object)
            if left_type == 'name':
                pronoun, address = decode_name(subexpressions[0], name_dict)
                pronouns = f'a{pronoun}a'
                operands = [address]
            elif left_type == 'number':
                number = subexpressions[0]
                pronouns = 'aca'
                operands = [number]
            else:
                raise Exception("This should never execute. Something is wrong")
        else:
            if left_type == 'name':
                pronoun, address = decode_name(subexpressions[0], name_dict)
                pronouns = f'a{pronoun}'
                operands = [address]
            elif left_type == 'number':
                number = subexpressions[0]
                pronouns = 'ac'
                operands = [number]
            else:
                raise Exception("This should never execute. Something is wrong")
            if right_type == 'name':
                pronoun, address = decode_name(subexpressions[1], name_dict)
                pronouns += pronoun
                operands += [address]
            elif right_type == 'number':
                number = subexpressions[1]
                pronouns += 'c'
                operands += [number]
            else:
                raise Exception("This should never execute. Something is wrong")
        
        assembly_mapping = {
            '+': 'add',
            '-': 'sub',
            '*': 'mul', # todo
            '/': 'div', # todo
            '%': 'mod', # todo
            '<<': 'shl', # todo
            '>>': 'shr', # todo
            '&': 'and',
            '^': 'xor',
            '|': 'or',
        }

        instr = assembly_mapping.get(operator)
        if instr:
            generated_code += instruction(instr+pronouns, operands=operands)
        
        elif operator in ['==', '!=', '<=', '>=', '<', '>'] and jumpy_task:
            generated_code += instruction('cmp'+pronouns, operands=operands)
            if reverse_jumpy_task:
                comparison = {
                    '==': 'je',
                    '!=': 'jne',
                    '<=': 'jng',
                    '>=': 'jnl',
                    '<': 'jl',
                    '>': 'jg',
                }
            else:
                comparison = {
                    '==': 'jne',
                    '!=': 'je',
                    '<=': 'jg',
                    '>=': 'jl',
                    '<': 'jnl',
                    '>': 'jng',
                }
            # jeśli nie udało się spełnić warunku (jeśli spełniono odwrotny), wtedy skok
            generated_code += instruction(comparison.get(operator), operands=[jumpy_task])
            return generated_code

        elif operator in ['==', '!=', '<=', '>=', '<', '>']:
            jump_if_true = jump_counter_object.pop()
            jump_counter_object.append(jump_if_true+1)
            # pronouns = pronouns[-2:]
            generated_code += instruction('cmp'+pronouns, operands=operands)
            # zakładamy że jest prawdziwe
            generated_code += instruction('movac', operands=[1])
            # jeśli rzeczywiście jest prawdziwe to unikamy wyzerowania tej prawdy
            comparison = {
                '==': 'je',
                '!=': 'jne',
                '<=': 'jng',
                '>=': 'jnl',
                '<': 'jl',
                '>': 'jg',
            }
            generated_code += instruction(comparison.get(operator), operands=[f'l{jump_if_true}'])
            # zerujemy prawdę jeśli się nie udało zrobić skoku
            generated_code += instruction('movac', operands=[0])
            # always
            generated_code += label(f'l{jump_if_true}')
            return generated_code
        
        else:
            raise Exception(f"Instrukcja jeszcze nie obsługiwana. Oto ekspresja: {expression}. Oto operator: {operator}")
        
        return generated_code
    
    
    if operator == 'addressof':
        assert number_of_subexpressions == 1, f'"{operator}" operator has a wrong number of subexpressions. Exprected 1, got {number_of_subexpressions}'
        left_operator, left_subexpressions = parse_expression(subexpressions[0])
        if classify_c_string(subexpressions[0]) == 'name':
            variable_name = subexpressions[0]
            pronoun, address = decode_name(variable_name, name_dict)
            if pronoun == 'e':
                generated_code += instruction('movaf', comment='przenosimy wartość frame do akumulatora')
                generated_code += instruction('addac', operands=[address], comment=f"przesuwamy wartość frame w ret o odległość '{variable_name}' żeby uzyskać wskaźnik na {variable_name}")
                return generated_code
            else:
                generated_code += instruction('movac', operands=[address])
                return generated_code
        elif left_operator == 'dereference':
            generated_code += evaluate_expression(left_subexpressions[0], name_dict, machine, jump_counter_object)
            return generated_code
        elif left_operator == 'index':
            generated_code += evaluate_expression("+".join(left_subexpressions), name_dict, machine, jump_counter_object)
            return generated_code
        else:
            raise ValueError(f"addressof can only be run on variables, dereferences or indexes. Got {subexpressions[0]}. The entire expression is: {expression}")
            
    
    if operator == 'dereference':
        assert number_of_subexpressions == 1, f'"{operator}" operator has a wrong number of subexpressions. Exprected 1, got {number_of_subexpressions}'
        if classify_c_string(subexpressions[0]) == 'name' and machine.get('eac'):
            pronoun, address = decode_name(subexpressions[0], name_dict)
            generated_code += instruction(f'mova{pronoun}', operands=[address])
            generated_code += instruction('movad')
            # this never exacutes
        else:
            generated_code += evaluate_expression(subexpressions[0], name_dict, machine, jump_counter_object)
            generated_code += instruction('movad', comment="przenosimy to co pod adresem akumulatora do akumulatora")
        return generated_code

    
    if operator in ['cmp_varpp', 'cmp_ppvar', 'cmp_varmm', 'cmp_mmvar']:
        left_operator, left_subexpressions = parse_expression(subexpressions[0])
        assert len(left_subexpressions) == 1, "Only dereferences or variables can be assigned"
        assert classify_c_string(left_subexpressions[0]) == 'name' or left_operator == 'dereference', \
            f'{operator} can only be used on variables or dereferences'

        if classify_c_string(left_subexpressions[0]) == 'name' and machine.get('eac'):
            pronoun, address = decode_name(left_subexpressions[0], name_dict)
            if operator == 'cmp_varpp':
                if not is_top_level:
                    generated_code += instruction(f'mova{pronoun}', operands=[address])
                generated_code += instruction(f'add{pronoun}c', operands=[address, 1])
            elif operator == 'cmp_ppvar':
                generated_code += instruction(f'add{pronoun}c', operands=[address, 1])
                if not is_top_level:
                    generated_code += instruction(f'mova{pronoun}', operands=[address])
            elif operator == 'cmp_varmm':
                if not is_top_level:
                    generated_code += instruction(f'mova{pronoun}', operands=[address])
                generated_code += instruction(f'sub{pronoun}c', operands=[address, 1])
            elif operator == 'cmp_mmvar':
                generated_code += instruction(f'sub{pronoun}c', operands=[address, 1])
                if not is_top_level:
                    generated_code += instruction(f'mova{pronoun}', operands=[address])
            return generated_code
        
        elif classify_c_string(left_subexpressions[0]) == 'name':
            generated_code += evaluate_expression("addressof(" + subexpressions[0] + ")", name_dict, machine, jump_counter_object)
        elif left_operator == 'dereference':
            generated_code += evaluate_expression(left_subexpressions[0], name_dict, machine, jump_counter_object)
        else:
            raise Exception('Unexpected something')

        if is_top_level:
            if operator == 'cmp_varpp':
                generated_code += instruction('adddc', operands=[1])
            elif operator == 'cmp_ppvar':
                generated_code += instruction('adddc', operands=[1])
            elif operator == 'cmp_varmm':
                generated_code += instruction('subdc', operands=[1])
            elif operator == 'cmp_mmvar':
                generated_code += instruction('subdc', operands=[1])
            return generated_code
        
        if operator == 'cmp_varpp':
            generated_code += instruction('movbd')
            generated_code += instruction('adddc', operands=[1])
            generated_code += instruction('movab')
        elif operator == 'cmp_ppvar':
            generated_code += instruction('adddc', operands=[1])
            generated_code += instruction('movad')
        elif operator == 'cmp_varmm':
            generated_code += instruction('movbd')
            generated_code += instruction('subdc', operands=[1])
            generated_code += instruction('movab')
        elif operator == 'cmp_mmvar':
            generated_code += instruction('subdc', operands=[1])
            generated_code += instruction('movad')
        return generated_code

    
    if operator == '&':
        raise Exception(f'Maszyna nie posiada bitwise and. Oto ekspresja: {expression}')

    
    if operator == '|':
        raise Exception(f'Maszyna nie posiada bitwise or. Oto ekspresja: {expression}')

    
    if operator == '^':
        raise Exception(f'Maszyna nie posiada bitwise xor. Oto ekspresja: {expression}')
    
    
    if operator == '&&':
        jump_if_false = jump_counter_object.pop()
        jump_if_true = jump_if_false+1
        jump_counter_object.append(jump_if_true+1)

        generated_code += evaluate_expression(subexpressions[0], name_dict, machine, jump_counter_object)
        generated_code += instruction('cmpac', operands=[0])
        generated_code += instruction('je', operands=[f'l{jump_if_false}'])
        
        generated_code += evaluate_expression(subexpressions[1], name_dict, machine, jump_counter_object)
        generated_code += instruction('cmpac', operands=[0])
        generated_code += instruction('je', operands=[f'l{jump_if_false}'])

        # if true
        generated_code += instruction('movac', operands=[1])
        generated_code += instruction('jmp', operands=[f'l{jump_if_true}'])
        
        # if false
        generated_code += label(f'l{jump_if_false}')
        generated_code += instruction('movac', operands=[0])

        # always
        generated_code += label(f'l{jump_if_true}')
        return generated_code

    
    if operator == '||':
        jump_if_true = jump_counter_object.pop()
        jump_if_false = jump_if_true+1
        jump_always = jump_if_false+1
        jump_counter_object.append(jump_always+1)

        generated_code += evaluate_expression(subexpressions[0], name_dict, machine, jump_counter_object)
        generated_code += instruction('cmpac', operands=[0])
        generated_code += instruction('jne', operands=[f'l{jump_if_true}'])

        generated_code += evaluate_expression(subexpressions[1], name_dict, machine, jump_counter_object)
        generated_code += instruction('cmpac', operands=[0])
        generated_code += instruction('je', operands=[f'l{jump_if_false}'])

        # if true
        generated_code += label(f'l{jump_if_true}')
        generated_code += instruction('movac', operands=[1])
        generated_code += instruction('jmp', operands=[f'l{jump_always}'])

        # if false
        generated_code += label(f'l{jump_if_false}')
        generated_code += instruction('movac', operands=[0])

        # always
        generated_code += label(f'l{jump_always}')
        return generated_code
        
    
    # operator-function mapping
    operator_function_map = {
        '*': 'mul',
        '%': 'mod',
        '/': 'div',
    }
    operator = operator_function_map.get(operator) if operator in operator_function_map else operator


    # funkcje

    for i in range(number_of_subexpressions-1, -1, -1):
        variable_name = subexpressions[i]
        if classify_c_string(variable_name) == 'number':
            generated_code += instruction('pushc', operands=[variable_name])
        elif classify_c_string(variable_name) == 'name' and machine.get('eac'):
            pronoun, address = decode_name(variable_name, name_dict)
            generated_code += instruction(f'push{pronoun}', operands=[address])
        else:
            generated_code += evaluate_expression(subexpressions[i], name_dict, machine, jump_counter_object)
            generated_code += instruction('pusha')

    generated_code += instruction('call', operands=[operator])
    if number_of_subexpressions > 0:
        generated_code += instruction('subsc', operands=[number_of_subexpressions])

    return generated_code


def print_instructions(instructions):
    kompy = None
    for instruction in instructions:
        if instruction.get('type') == 'comment':
            kompy = instruction.get('comment')
            continue
            print(f"# {instruction.get('comment')}")
        if instruction.get('type') == 'instruction':
            t_instruction = instruction.get('instruction')
            # t_operands = " ".join(instruction.get('operands'))
            t_operands = " ".join([f"{i}" for i in instruction.get('operands')])
            t_comment = instruction.get('comment')
            t_comment = kompy
            if t_comment:
                print(f"\t{t_instruction}\t{t_operands}\t{t_comment}")
            else:
                print(f"\t{t_instruction}\t{t_operands}")
        if instruction.get('type') == 'label':
            t_label = instruction.get('label')
            print(f'{t_label}:')


"""name_dict = {
    'a': 5,
    'b': 6,
    'c': 7,
    'd': 8,
    'x': 9,
}

machine = {
    'eac': True,
}

print_instructions(evaluate_expression("a + b", name_dict, machine))
print_instructions(evaluate_expression("(*c) = a + b", name_dict, machine))
exit()"""


def compile(function_name, function, arguments, machine, global_name_dict: None|dict=None):
    print('function:', function)
    print("these are the function arguments:", arguments)
    generated_code = []
    control_stack = [] # stack może być równy None, 'while', 'do', 'if', 'for' lub może zawierać metadane skoków
    # while zawiera jeden argument w lokalizacji -1 będący adresem skoku po skończeniu while
    jump_counter_object = [0]
    last_if_or_while = None
    expecting_while_after_do = None
    expecting_a_for_expression = None

    name_dict = {}
    if global_name_dict:
        name_dict = global_name_dict.copy()
    name_pointer = -3 # -1 and -2 are for the call function
    for argument in arguments:
        name_dict.update({argument: name_pointer})
        name_pointer -= 1
    name_pointer = 0

    for statement in function:
        print('Statement:', statement)
        generated_code += comment(f'Statement: {statement}')

        if expecting_a_for_expression == 'for':
            expression = '(' + statement.replace(';', ')')
            # inicjalizacja
            generated_code += label(f'l{jump_counter_object[0]}')
            new_l = jump_counter_object[0] + 1
            # generowanie kodu
            jump_counter_object = [jump_counter_object[0] + 2]
            # top level optimalization
            operator, _ = parse_expression(expression)
            if operator in ['==', '!=', '<=', '>=', '<', '>']:
                generated_code += evaluate_expression(expression, name_dict, machine, jump_counter_object, jumpy_task=f'l{new_l}')
            # not opimized
            else:
                generated_code += evaluate_expression(expression, name_dict, machine, jump_counter_object)
                generated_code += instruction('cmpac', operands=[0])
                generated_code += instruction('je', operands=[f'l{new_l}'])
            expecting_a_for_expression = 'last'
            continue

        if expecting_a_for_expression == 'last':
            # zakończenie
            control_stack.append(new_l)
            control_stack.append('(' + statement)
            control_stack.append("for")
            expecting_a_for_expression = None
            continue

        if (expecting_while_after_do is not None) and 'while' in statement:
            expression = re.search(r'\([^()]*\)', statement).group() # wykrywanie
            # top level optimalization
            operator, _ = parse_expression(expression)
            if operator in ['==', '!=', '<=', '>=', '<', '>']:
                generated_code += evaluate_expression(expression, name_dict, machine, jump_counter_object, jumpy_task=f'l{expecting_while_after_do}', reverse_jumpy_task=True)
            # not opimized
            else:
                generated_code += evaluate_expression(expression, name_dict, machine, jump_counter_object)
                generated_code += instruction('cmpac', operands=[0])
                generated_code += instruction('jne', operands=[f'l{expecting_while_after_do}'])
            expecting_while_after_do = None
            continue
        
        if expecting_while_after_do is not None:
            continue

        if statement == '{':
            control_stack.append(None)
            continue
        
        elif statement == '}':
            control_stack.pop(-1)
        
        elif 'while' in statement: # jump: str, id = 'while': str
            # wykrywanie
            expression = re.search(r'\([^()]*\)', statement).group()
            # inicjalizacja
            generated_code += label(f'l{jump_counter_object[0]}')
            new_l = jump_counter_object[0] + 1
            # generowanie kodu
            jump_counter_object = [jump_counter_object[0] + 2]
            # top level optimalization
            operator, _ = parse_expression(expression)
            if operator in ['==', '!=', '<=', '>=', '<', '>']:
                generated_code += evaluate_expression(expression, name_dict, machine, jump_counter_object, jumpy_task=f'l{new_l}')
            # not opimized
            else:
                generated_code += evaluate_expression(expression, name_dict, machine, jump_counter_object)
                generated_code += instruction('cmpac', operands=[0])
                generated_code += instruction('je', operands=[f'l{new_l}'])
            # zakończenie
            control_stack.append(new_l)
            control_stack.append("while")
            continue
        
        elif 'for' in statement:
            # wykrywanie
            stat = statement.replace(';', ')').replace(" ", "") # żeby wykrywacz ekspresji wykrył ekspresję
            expression = re.search(r'\([^()]*\)', stat).group()
            # ewaliuacja
            generated_code += evaluate_expression(expression, name_dict, machine, jump_counter_object, is_top_level=True)
            expecting_a_for_expression = 'for'
            continue
        
        elif 'else' in statement:
            # removing the last_if_or_while label, so that it can be placed later
            for instr in generated_code:
                if instr.get('label') == f'l{last_if_or_while}':
                    generated_code.remove(instr)
                    break
            generated_code += instruction('jmp', operands=[f'l{jump_counter_object[0]}'])
            generated_code += label(f'l{last_if_or_while}')
            control_stack.append(jump_counter_object[0])
            control_stack.append("if_or_else") # else ma identyczne zakonczenie co if
            jump_counter_object = [jump_counter_object[0] + 1]
            continue
        
        elif 'do' in statement:
            generated_code += label(f'l{jump_counter_object[0]}')
            control_stack.append(jump_counter_object[0])
            control_stack.append("do")
            jump_counter_object = [jump_counter_object[0] + 1]
            continue
        
        elif 'if' in statement:
            # wykrywanie
            expression = re.search(r'\([^()]*\)', statement).group()
            # inicjalizacja
            jump_ifa = jump_counter_object[0]
            jump_counter_object = [jump_counter_object[0] + 1]
            # top level optimalization
            operator, _ = parse_expression(expression)
            if operator in ['==', '!=', '<=', '>=', '<', '>']:
                generated_code += evaluate_expression(expression, name_dict, machine, jump_counter_object, jumpy_task=f'l{jump_ifa}')
            # not opimized
            else:
                generated_code += evaluate_expression(expression, name_dict, machine, jump_counter_object)
                generated_code += instruction('cmpac', operands=[0])
                generated_code += instruction('je', operands=[f'l{jump_ifa}'])
            # zakończenie
            control_stack.append(jump_ifa)
            control_stack.append("if_or_else")
            continue
        
        elif 'return' in statement:
            statement = statement.replace('return', '')
            generated_code += evaluate_expression(statement[:-1], name_dict, machine, jump_counter_object) # pozbywamy się ostatniego średnika
            generated_code += instruction('popf')
            generated_code += instruction('ret')


        # deklaracje
        elif re.match(r'^\s*(?:int|float|double|char|void|long|short|unsigned|signed)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;', statement) is not None:
            name = re.search(r'^\s*(?:int|float|double|char|void|long|short|unsigned|signed)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;', statement).group(1)
            name_dict.update({name: name_pointer})
            name_pointer += 1
        
        # samotne ekspresje
        else:
            generated_code += evaluate_expression(statement[:-1], name_dict, machine, jump_counter_object, is_top_level=True) # pozbywamy się ostatniego średnika
        

        if len(control_stack) == 0:
            generated_code += instruction('popf')
            generated_code += instruction('ret')
            append_to_generated_code = label(function_name)
            append_to_generated_code += instruction('pushf', comment="pushujemy stary wskaznik ramki")
            append_to_generated_code += instruction('movfs', comment="aktualizujemy wskaźnik okna na wartość wskaźnika staka")
            if name_pointer:
                append_to_generated_code += instruction('addsc', operands=[name_pointer], comment="zwiększamy stak o liczbę zadeklarowanych zmiennych")
            generated_code = append_to_generated_code + generated_code
            return generated_code # wtedy robimy return

        if control_stack[-1] == 'while':
            # inicjalizacja
            control_stack.pop(-1)
            new_l = control_stack.pop(-1)
            # generowanie kodu
            old_l = new_l - 1
            generated_code += instruction('jmp', operands=[f'l{old_l}'])
            generated_code += label(f'l{new_l}')
            last_if_or_while = new_l
            continue
        
        elif control_stack[-1] == 'if_or_else':
            control_stack.pop(-1)
            l = control_stack.pop(-1)
            generated_code += label(f'l{l}')
            last_if_or_while = l
            continue
        
        elif control_stack[-1] == 'do':
            control_stack.pop(-1)
            expecting_while_after_do = control_stack.pop(-1)
            continue

        elif control_stack[-1] == 'for':
            control_stack.pop(-1)
            expression = control_stack.pop(-1)
            generated_code += comment(f'Statement: {expression}')
            generated_code += evaluate_expression(expression, name_dict, machine, jump_counter_object, is_top_level=True)
            new_l = control_stack.pop(-1)
            old_l = new_l - 1
            generated_code += instruction('jmp', operands=[f'l{old_l}'])
            generated_code += label(f'l{new_l}')
            continue
    
    print('To nie powinno tu dojść nigdy')

# -----------------------------------------------------------------------------
# Chapter 3 - the rest and examples
# -----------------------------------------------------------------------------

# Example usage with control statements
code = """
int main() {
    int x = 5;
    if (x > 0)
        x++;
    else
        x--;
    do
        {x = x + 1;}
    while (x < 10);
    for (int i = 0; i < 5; i++)
        x += i; // x = x + i;
    return x;
}
"""

codea = """
int main() {
    int x = 5;
}"""

codef = """
int fibon(int n) {
    int a = 0;
    int b = 1;
    int temp;
    int i = 0;
    while (i < n) {
        temp = a+b;
        a = b;
        b = temp;
        i++;
    }
    return a;
}
"""

codea = """
void a() {
    int a;
    int b;
    int c = a || b;
    c = c == b;
}
"""

code = """
bool isPrime(int n) {
    // Handle edge cases
    if (n <= 1) return false; 
    if (n == 2 || n == 3) return true;

    // Check divisibility up to the square root of n
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return false;
    }

    return true;  // If no divisors were found, it's prime
}"""

codes = """
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    temp = temp;
}
"""

codeg = """
void insertion(int *list, int size) {
    for (int i = 1; i < size; i++) {
        for (int j = i-1; j >= 0; j--) {
            if (list[j] > list[j+1]) {
                swap(&list[j], &list[j+1]);
            }
        }
    }
}
"""

codea = """
void generate_random_list(int *list, int size) {
    for (int i = 0; i < size; i++) {
        list[i] = rand() % 1000; // Random numbers between 0 and 999
    }
}
"""

codea = """
void return_a_global() {
    return a;
}
"""

codem = """
int mul(int a, int b) {
    int result = 0;
    while (b > 0) {
        result += a;
        b--;
    }
    return result;
}
"""

codeslide = """
void slide(int *list, int len) {
    int temp = list[len-1];
    for (int i = len-2; i >= 0; i--) {
        list[i+1] = list[i];
    }
    list[0] = temp;
}
"""

code = codeslide

functions_list = extract_functions(code)
# Output the list of functions
print(functions_list)

def preprocess(code):
    print("code", code)
    lines = truncate_to_expressions(code)
    lines = simplify_declarations(lines)
    lines = simplify_for_loop_declarations(lines)
    lines = replace_true_false(lines)
    lines = replace_increment_decrement(lines)
    print('lines:', lines)
    return lines

machine = {
    'eac': True,
}

global_name_dict = {
    'a': {'type': 'global', 'address': 4},
}

print('Compiling the following code:')
print(code)
print()
print('List of functions:', [parse_function(function)[0] for function in functions_list])

for function in functions_list:
    funky = parse_function(function)
    print(f"\nCompiling {funky[0]}")
    print_instructions(compile(funky[0], preprocess("{" + funky[2] + "}"), funky[1], machine))

