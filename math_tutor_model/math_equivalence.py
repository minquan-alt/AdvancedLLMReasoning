# Source: https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
# We extend the official MATH equivalence checker with the following normalizations:

# 1. Format standardization:
#    - Complex number notation: I ↔ i
#    - Power notation: ** ↔ ^
#    - Implicit multiplication: 2*k ↔ 2k
#    - Whitespace removal

# 2. LaTeX expression handling:
#    - Convert LaTeX sqrt to Python evaluable form
#    - Support nested sqrt expressions

# 3. Numeric tolerance:
#    - Float equality with ε = 1e-9
#    - Integer normalization (5.0 → 5)

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    
    # ============== my addition ============: fix compare int and float strings, fix latex sqrt
    import re
    
    # Simple normalization for common format differences
    str1_norm = str(str1).strip()
    str2_norm = str(str2).strip()
    
    # Normalize complex numbers: I -> i, *I -> i, *i -> i
    str1_norm = re.sub(r'\*I\b', 'i', str1_norm)
    str1_norm = re.sub(r'\*i\b', 'i', str1_norm)
    str1_norm = re.sub(r'\bI\b', 'i', str1_norm)
    str2_norm = re.sub(r'\*I\b', 'i', str2_norm)
    str2_norm = re.sub(r'\*i\b', 'i', str2_norm)
    str2_norm = re.sub(r'\bI\b', 'i', str2_norm)
    
    # Normalize power notation: ** -> ^
    str1_norm = str1_norm.replace('**', '^')
    str2_norm = str2_norm.replace('**', '^')
    
    # Remove explicit multiplication before variables: 2*k -> 2k, 3*x -> 3x
    str1_norm = re.sub(r'(\d)\*([a-zA-Z])', r'\1\2', str1_norm)
    str2_norm = re.sub(r'(\d)\*([a-zA-Z])', r'\1\2', str2_norm)
    
    # Remove spaces
    str1_norm = re.sub(r'\s+', '', str1_norm)
    str2_norm = re.sub(r'\s+', '', str2_norm)
    
    # Quick check after normalization
    if str1_norm == str2_norm:
        return True
    
    try:
        num1 = float(str1_norm)
        num2 = float(str2_norm)
        if abs(num1 - num2) < 1e-9:
            return True
        if num1 == int(num1):
            str1_norm = str(int(num1))
        if num2 == int(num2):
            str2_norm = str(int(num2))
    except:
        pass
    
    try:
        import math
        eval_str1 = str1_norm
        eval_str2 = str2_norm
        
        def replace_latex_frac(s):
            # Convert \frac{a}{b} to (a)/(b), handling nested braces
            s = re.sub(r'\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'(\1)/(\2)', s)
            return s
        
        def replace_latex_sqrt(s):
            s = re.sub(r'\\sqrt\{([^}]+)\}', r'math.sqrt(\1)', s)
            s = re.sub(r'(\d)math\.sqrt', r'\1*math.sqrt', s)
            s = re.sub(r'\)math\.sqrt', r')*math.sqrt', s)
            return s
        
        # First convert fractions, then sqrt
        eval_str1 = replace_latex_frac(eval_str1)
        eval_str2 = replace_latex_frac(eval_str2)
        
        eval_str1 = replace_latex_sqrt(eval_str1)
        eval_str2 = replace_latex_sqrt(eval_str2)
        
        eval_str1 = re.sub(r'(?<!math\.)sqrt\(', r'math.sqrt(', eval_str1)
        eval_str2 = re.sub(r'(?<!math\.)sqrt\(', r'math.sqrt(', eval_str2)
        
        # Add * before math.sqrt if missing
        eval_str1 = re.sub(r'(\d)math\.sqrt', r'\1*math.sqrt', eval_str1)
        eval_str2 = re.sub(r'(\d)math\.sqrt', r'\1*math.sqrt', eval_str2)
        
        # Convert ^ back to ** for eval
        eval_str1 = eval_str1.replace('^', '**')
        eval_str2 = eval_str2.replace('^', '**')
        
        eval_str1 = eval_str1.replace('\\', '')
        eval_str2 = eval_str2.replace('\\', '')
        
        val1 = eval(eval_str1, {"math": math, "__builtins__": {}})
        val2 = eval(eval_str2, {"math": math, "__builtins__": {}})
        if abs(val1 - val2) < 1e-9:
            return True
    except:
        pass
    # ============== my addition ============
    try:
        ss1 = _strip_string(str1_norm)
        ss2 = _strip_string(str2_norm)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1_norm == str2_norm