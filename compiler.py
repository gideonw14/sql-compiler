""" SPI - Simple Pascal Interpreter. Part 9."""

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis
INTEGER         = 'INTEGER'
STRING          = 'STRING'
PLUS            = 'PLUS'
MINUS           = 'MINUS'
MUL             = 'MUL'
LPAREN          = 'LPAREN'
RPAREN          = 'RPAREN'
ID              = 'ID'
ASSIGN          = 'ASSIGN'
SEMI            = 'SEMI'
DOT             = 'DOT'
COLON           = 'COLON'
COMMA           = 'COMMA'
EOF             = 'EOF'
KEYWORD         = 'KEYWORD'
SELECT          = 'SELECT'
FROM            = 'FROM'
WHERE           = 'WHERE'
AS              = 'AS'
IN              = 'IN'
INTERSECT       = 'INTERSECT'
UNION           = 'UNION'
EXCEPT          = 'EXCEPT'
HAVING          = 'HAVING'
GROUPBY         = 'GROUPBY'
AND             = 'AND'
OR              = 'OR'
EQUAL           = 'EQUAL'
GREATER         = 'GREATER'
LESSER          = 'LESSER'
GREATEREQUAL    = 'GREATEREQUAL'
LESSEREQUAL     = 'LESSEREQUAL'

SPACES = 8
# Helper Function
def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

class Tree_Node(object):
    def __init__(self, left, right, value):
        self.left = left
        self.right = right
        self.value = value

class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        """String representation of the class instance.

        Examples:
            Token(INTEGER, 3)
            Token(PLUS, '+')
            Token(MUL, '*')
        """
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()


RESERVED_KEYWORDS = {
    'BEGIN': Token('BEGIN', 'BEGIN'),
    'END': Token('END', 'END'),
    'SELECT': Token('SELECT', 'SELECT'),
    'FROM': Token('FROM', 'FROM'),
    'WHERE': Token('WHERE', 'WHERE'),
    'AS': Token('AS', 'AS'),
    'AND': Token('AND', 'AND'),
    'OR': Token('OR', 'OR'),
    'IN': Token('IN', 'IN'),
    'INTERSECT': Token('INTERSECT', 'INTERSECT'),
    'UNION': Token('UNION', 'UNION'),
    'EXCEPT': Token('EXCEPT', 'EXCEPT'),
    'HAVING': Token('HAVING', 'HAVING'),
    'GROUPBY': Token('GROUPBY', 'GROUPBY'),

}


class Lexer(object):
    def __init__(self, text):
        # client string input, e.g. "4 + 2 * 3 - 6 / 2"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid character near or at "{}"'.format(self.current_char))

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def integer(self):
        """Return a (multidigit) integer consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def string(self):
        """ Return a string consumed from the input """
        result = ''
        if self.current_char == "'":
            self.advance()
        # import ipdb; ipdb.set_trace()
        while self.current_char != "'":
            result += str(self.current_char)
            self.advance()
        if self.current_char == "'":
            self.advance()
        else:
            self.error()
        return result

    def _id(self):
        """Handle identifiers and reserved keywords"""
        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()

        token = RESERVED_KEYWORDS.get(result, Token(ID, result)) # Gets the keyword or returns identifier token
        return token

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())

            if self.current_char == "'":
                return Token(STRING, self.string())

            if self.current_char == ':' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(ASSIGN, ':=')

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(DIV, '/')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

            if self.current_char == '=':
                self.advance()
                return Token(EQUAL, '=')

            if self.current_char == '>' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(GREATEREQUAL, '>=')

            if self.current_char == '<' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(LESSEREQUAL, '<=')

            if self.current_char == '>':
                self.advance()
                return Token(GREATER, '>')

            if self.current_char == '<':
                self.advance()
                return Token(LESSER, '<')

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')

            self.error()

        return Token(EOF, None)


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################

class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Compound(AST):
    """Represents a 'BEGIN ... END' block"""
    def __init__(self):
        self.children = []


class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Var(AST):
    """The Var node is constructed out of ID token."""
    def __init__(self, token):
        self.token = token
        self.value = token.value

class Rel_Alg_Select(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Attr(AST):
    def __init__(self, attribute, relation=None):
        self.attribute = attribute.value
        if relation:
            self.relation = relation.value
        else:
            self.relation = None

class Rel(AST):
    def __init__(self, relation, alias=None):
        self.relation = relation.value
        if alias:
            self.alias = alias.value
        else:
            self.alias = None

class Combo(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right


class NoOp(AST):
    pass


class In(AST):
    def __init__(self, attribute, select):
        pass

class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax near or at "{}"'.format(self.current_token.value))

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            print(self.current_token)
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def query(self):
        # query: compound statement;
        node = self.sql_compound_statement()
        # self.eat(SEMI)
        return node

    def sql_compound_statement(self):
        """
        note: ? means 0 or 1 instances
        sql_compound_statement: SELECT attribute_list
                                FROM relation_list
                                (WHERE condition_list)?
                                (GROUP BY attribute_list)?
                                (HAVING condition_list)?
                                (INTERSECT | UNION | EXCEPT sql_compound_statement)?
        """
        cond_nodes = None
        group_by_list = None
        having_list = None
        compound_statement = None
        set_op = None
        self.eat(SELECT)
        attr_nodes = self.attribute_list()
        self.eat(FROM)
        rel_nodes = self.relation_list()
        if self.current_token.type == WHERE:
            self.eat(WHERE)
            cond_nodes = self.condition_list()
        if self.current_token.type == GROUPBY:
            self.eat(GROUPBY)
            group_by_list = self.attribute_list()
        if self.current_token.type == HAVING:
            self.eat(HAVING)
            having_list = self.condition_list()
        if self.current_token.type in (INTERSECT, UNION, EXCEPT):
            set_op = Combo(None, self.current_token.value, None)
            if self.current_token.type == INTERSECT:
                self.eat(INTERSECT)
            elif self.current_token.type == UNION:
                self.eat(UNION)
            elif self.current_token.type == EXCEPT:
                self.eat(EXCEPT)
            compound_statement = self.sql_compound_statement()
        roots = list()
        root = Compound()
        for node in attr_nodes:
            root.children.append(node)
        for node in rel_nodes:
            root.children.append(node)
        if cond_nodes:
            for node in cond_nodes:
                root.children.append(node)
        if group_by_list:
            for node in group_by_list:
                root.children.append(node)
        if having_list:
            for node in having_list:
                root.children.append(node)
        if compound_statement:
            set_op.left = root
            set_op.right = compound_statement
            return set_op
        else:
            roots.append(root)
            return roots

    def attribute_list(self):
        """
        attribute_list : attribute
                       | attribute COMMA attribute_list
        """
        node = self.attribute()
        results = [node]
        while self.current_token.type == COMMA:
            self.eat(COMMA)
            results.append(self.attribute())
        return results

    def attribute(self):
        """
        attribute : identifier
                  | identifier DOT identifier
                  | STAR aka MUL

        """
        node = Attr(self.current_token)
        if self.current_token.type == MUL:
            self.eat(MUL)
        else:
            self.eat(ID)
            if self.current_token.type == DOT:
                self.eat(DOT)
                node.relation = node.attribute
                node.attribute = self.current_token.value
                self.eat(ID)
        return node

    def relation_list(self):
        """
        relation_list : relation
                      | relation COMMA relation_list
        """
        node = self.relation()
        results = [node]
        while self.current_token.type == COMMA:
            self.eat(COMMA)
            results.append(self.relation())
        return results

    def relation(self):
        """
        relation : identifier
                 | identifier AS identifier
        """
        node = Rel(self.current_token)
        self.eat(ID)
        if self.current_token.type == AS:
            self.eat(AS)
            node.alias = self.current_token.value
            self.eat(ID)
        return node

    def condition_list(self):
        """
        condition_list : condition
                       | condition (AND | OR) condition_list
        """
        node = self.condition()
        results = [node]
        while self.current_token.type in (AND, OR):
            if self.current_token.type == AND:
                self.eat(AND)
            else:
                self.eat(OR)
            results.append(self.condition())
        return results

    def condition(self):
        """
        condition : attribute (EQUAL | GREATER | LESSER | GREATEREQUAL | LESSEREQUAL) (attribute | INTEGER | STRING)
                  | attribute IN LPAREN sql_compound_statement RPAREN
        """
        # Left is always attribute
        left = self.attribute()
        if self.current_token.type in (EQUAL, GREATER, LESSER, GREATEREQUAL, LESSEREQUAL):
            # Comparison
            token = self.current_token
            if self.current_token.type == EQUAL:
                self.eat(EQUAL)
            elif self.current_token.type == GREATER:
                self.eat(GREATER)
            elif self.current_token.type == LESSER:
                self.eat(LESSER)
            elif self.current_token.type == GREATEREQUAL:
                self.eat(GREATEREQUAL)
            elif self.current_token.type == LESSEREQUAL:
                self.eat(LESSEREQUAL)
            else: # This *should* never happen
                self.error()

            # Right: integer, string, or attribute
            if self.current_token.type == INTEGER:
                right = self.current_token
                self.eat(INTEGER)
            elif self.current_token.type == STRING:
                right = self.current_token
                self.eat(STRING)
            else: # attribute
                right = self.attribute()
            return Rel_Alg_Select(left, token, right)
        elif self.current_token.type == IN:
            self.eat(IN)
            self.eat(LPAREN)
            node = self.sql_compound_statement()
            right = node[0].children.pop(0) # only 1 attribute on IN
            self.eat(RPAREN)
            node[0].children.append(Rel_Alg_Select(left, Token(EQUAL, '='), right))
            return node

    def parse_sql(self):
        """
        query: sql_compound_statement
        sql_compound_statement: SELECT attributes FROM (relations | query) WHERE (conditions | attributes IN query)
        """
        node = self.query()
        if self.current_token.type != EOF:
            self.error()

        return node


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################

class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


class Query(object):
    def __init__(self, selects=list(), projects=list(), cross=list()):
        self.selects = selects
        self.projects = projects
        self.cross = cross

class Union_Inter_Diff(object):
    def __init__(self, query1, query2, op):
        self.query1 = query1
        self.query2 = query2
        self.op = op

class Interpreter(NodeVisitor):

    GLOBAL_SCOPE = {}
    QUERIES = list() # list of queries
    UNION_INTER_DIFFS = list() # list of unions/intersections/differences

    def __init__(self, parser):
        self.parser = parser

    def visit_Rel_Alg_Select(self, node):
        if node.left.relation: # always attribute
            left = node.left.relation + '.' + node.left.attribute
        else:
            left = node.left.attribute

        if isinstance(node.right, Attr):
            if node.right.relation:
                right = node.right.relation + '.' + node.right.attribute
            else:
                right = node.right.attribute
        else:
            right = str(node.right.value)
        result = left +' '+ node.op.value +' '+ right
        self.QUERIES[-1].selects.append(result)

    def visit_list(self, node):
        for item in node:
            self.visit(item)

    def visit_Num(self, node):
        return node.value

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == PLUS:
            return +self.visit(node.expr)
        elif op == MINUS:
            return -self.visit(node.expr)

    def visit_Compound(self, node):
        self.QUERIES.append(Query())
        index = len(self.QUERIES) - 1
        for child in node.children:
            self.visit(child)
        return index

    def visit_Assign(self, node):
        var_name = node.left.value
        self.GLOBAL_SCOPE[var_name] = self.visit(node.right)

    def visit_Var(self, node):
        var_name = node.value
        val = self.GLOBAL_SCOPE.get(var_name)
        if val is None:
            raise NameError(repr(var_name))
        else:
            return val

    def visit_Attr(self, node):
        atr_name = node.attribute
        if node.relation:
            rel_name = node.relation
            atr_name = rel_name + '.' + atr_name
        self.QUERIES[-1].projects.append(atr_name)

    def visit_Rel(self, node):
        rel_name = list()
        rel_name.append(node.relation)
        if node.alias:
            rel_name.append(node.alias)
        self.QUERIES[-1].cross.append(rel_name)

    def visit_Combo(self, node):
        import ipdb; ipdb.set_trace()
        left = self.visit(node.left)
        right = self.visit(node.right)
        self.UNION_INTER_DIFFS.append(Union_Inter_Diff(left, right, node.op))

    def visit_NoOp(self, node):
        pass


    def interpret(self):
        tree = self.parser.parse_sql()
        if tree is None:
            return ''
        return self.visit(tree)

def print_rel_alg(interpreter):
    print('######################################')
    print('#          Relation Algebra          #')
    print('######################################\n')
    print('PROJECT [', end='')
    for idx, item in enumerate(interpreter.PROJECTS):
        if idx == len(interpreter.PROJECTS) - 1:
            print(item, end='')
        else:
            print('{}, '.format(item), end='')
    print('] (SELECT [', end='')
    for idx, item in enumerate(interpreter.SELECTS):
        if idx == len(interpreter.SELECTS) - 1:
            print(item, end='')
        else:
            print('{}, '.format(item), end='')
    print('] (', end='')
    for idx, list in enumerate(interpreter.CROSS_PRODUCTS):
        if idx == len(interpreter.CROSS_PRODUCTS) - 1:
            if len(list) == 1:
                print(list[0], end='')
            else:
                print('{} AS {}'.format(list[0], list[1]), end='')
        else:
            if len(list) == 1:
                print('{} X '.format(list[0]), end='')
            else:
                print('{} AS {} X '.format(list[0], list[1]), end='')
    print(')))\n')

def build_query_tree(interpreter):
    project = 'PROJECT ['
    for idx, item in enumerate(interpreter.PROJECTS):
        if idx == len(interpreter.PROJECTS) - 1:
            project += item
        else:
            project += '{}, '.format(item)
    project += ']'
    tree = Tree_Node(None, None, project)
    select = 'SELECT ['
    for idx, item in enumerate(interpreter.SELECTS):
        if idx == len(interpreter.SELECTS) - 1:
            select += item
        else:
            select += '{}, '.format(item)
    select += ']'
    select_node = Tree_Node(None, None, select)
    tree.left = select_node
    cross_node = build_cross_tree(interpreter.CROSS_PRODUCTS)
    tree.left.left = cross_node
    return tree

def build_cross_tree(cross_prods):
    node = Tree_Node(None, None, None)
    if len(cross_prods) == 1:
        if len(cross_prods[0]) == 1:
            node.value = cross_prods[0][0]
        else:
            node.value = '{} AS {}'.format(cross_prods[0][0], cross_prods[0][1])
        return node
    elif len(cross_prods) == 2:
        right_node = Tree_Node(None, None, None)
        left_node = Tree_Node(None, None, None)
        right = cross_prods[1]
        left = cross_prods[0]
        if len(right) == 1:
            right_node.value = right[0]
        else:
            right_node.value = '{} AS {}'.format(right[0], right[1])
        if len(left) == 1:
            left_node.value = left[0]
        else:
            left_node.value = '{} AS {}'.format(left[0], left[1])
        node.left = left_node
        node.right = right_node
        node.value = 'X'
        return node
    else:
        right = cross_prods.pop(0)
        right_node = Tree_Node(None, None, None)
        if len(right) == 1:
            right_node.value = right[0]
        else:
            right_node.value = '{} AS {}'.format(right[0], right[1])
        node.right = right_node
        node.left = build_cross_tree(cross_prods)
        node.value = 'X'
        return node

def print_query_tree(tree, spaces):
    if tree:
        spaces += SPACES
        print_query_tree(tree.right, spaces)
        spaces -= SPACES
        if tree.right != None:
            print(' ' * spaces, end='')
            print('/')
        if spaces != 0:
            print(' '*(spaces - SPACES), end='')
            print(' |' + '-'*(SPACES-2), end='')
        print(tree.value)
        if tree.left != None:
            print(' ' * spaces, end='')
            print('\\')
        spaces += SPACES
        print_query_tree(tree.left, spaces)
        spaces -= SPACES
    return

def main():
    import sys
    text = open(sys.argv[1], 'r').read()
    tables = open(sys.argv[2], 'r').read()

    text = text.upper()
    lexer = Lexer(text)
    parser = Parser(lexer)
    interpreter = Interpreter(parser)
    result = interpreter.interpret()
    import ipdb; ipdb.set_trace()
    # print_rel_alg(interpreter)
    # tree = build_query_tree(interpreter)
    # print('######################################')
    # print('#            Query Tree              #')
    # print('######################################\n')
    # print_query_tree(tree, 0)


if __name__ == '__main__':
    main()
