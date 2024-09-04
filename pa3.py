# Name: pa3.py
# Author: Rodolfo Lopez
# Date: 11/12/21
# Description: 1: RE -> NFA, 2: NFA -> DFA, 3: Simulate DFA


class InvalidExpression(Exception):
    pass


class InvalidAlphabet(Exception):
    pass


class RegEx:
    operators = "* |"  # class variable to recog valid re operators

    # pa4 need to modify constructor to take in self.reg_exp as a parameter and and self.alpha as a parameter
    def __init__(self, filename):
        """
        Initializes a RegEx object from the specifications
        in the file whose name is filename.
        """
        infile = open(filename, "r")

        self.alpha = infile.readline().strip()

        if ("*" or "|" or "e" or "N") in self.alpha:  # ensure valid alphabet
            raise InvalidAlphabet()

        self.regexp = infile.readline().strip()  # read re

        infile.close()

    def simulate(self, str):
        """
        Returns True if the string str is in the language of
        the "self" regular expression.
        """

        # Step1
        root = self.toSyntaxTree()  # re -> abstract syntax tree
        regExpNFA = self.syntaxTreeToNFA(root)  # abstract syntax tree -> NFA

        # Step2
        regExpDFA = regExpNFA.toDFA()  # NFA -> DFA

        # Step3
        return regExpDFA.simulate(str)  # simulate DFA

    def isOperand(self, sym):
        """
        Returns true if symbol is operand
        """
        if sym == " ":
            return False
        elif sym in self.alpha:
            return True
        elif sym == "e" or sym == "N":
            return True
        return False

    def isOperator(self, sym):
        """
        Returns true if symbol is operator
        """
        if sym in RegEx.operators:
            return True
        return False

    def getOperatorPrec(self, operator):
        """
        Returns operator precedence of operator
        """
        if operator == "|":
            return 1
        elif operator == " ":
            return 2
        elif operator == "*":
            return 3
        return -1

    def addImpliedConcat(self):
        """
        Void method modifies regexp to recognize concatenation
        """
        self.regexp = self.regexp.replace(" ", "")
        prev = " "
        i = -1

        for sym in self.regexp:
            i += 1  # first sym
            if prev == " ":
                prev = sym
                continue
            # check each case that will result in concatenation
            elif (
                (self.isOperand(prev) and self.isOperand(sym))
                or (self.isOperand(prev) and sym == "(")
                or (prev == "*" and self.isOperand(sym))
                or (prev == "*" and sym == "(")
                or (prev == ")" and self.isOperand(sym))
                or (prev == ")" and sym == "(")
            ):
                self.regexp = self.regexp[:i] + " " + self.regexp[i:]
                i += 1  # accounts for space added
            prev = sym

    def popAndPushTree(self):
        """Pop operator and operand(s), create/push tree"""
        operand_right_node = None
        operand_left_node = None
        # pop
        if self.operand_stack:
            operand_right_node = self.operand_stack.pop()
        else:
            raise InvalidExpression()

        if self.operator_stack:
            operator_node = self.operator_stack.pop()
        else:
            raise InvalidExpression()

        # two pops if not unary operator
        if operator_node != "*":
            if self.operand_stack:
                operand_left_node = self.operand_stack.pop()

        # check operater node
        if operator_node == "|":
            if operand_left_node == "N":
                self.operand_stack.append(BinTree(operand_right_node))
            elif operand_right_node == "N":
                self.operand_stack.append(BinTree(operand_left_node))
            else:  # normal case
                self.operand_stack.append(
                    BinTree(operator_node, operand_left_node, operand_right_node)
                )

        elif operator_node == " ":
            if (operand_left_node == "N") or (operand_right_node == "N"):
                self.operand_stack.append(BinTree("N"))
            elif operand_left_node == "e":
                self.operand_stack.append(BinTree(operand_right_node))
            elif operand_right_node == "e":
                self.operand_stack.append(BinTree(operand_left_node))
            else:  # normal case
                self.operand_stack.append(
                    BinTree(operator_node, operand_left_node, operand_right_node)
                )

        elif operator_node == "*":
            if operand_left_node == "N":
                self.operand_stack.append(BinTree("e"))
            else:  # normal case
                self.operand_stack.append(BinTree(operator_node, operand_right_node))
        else:
            raise InvalidExpression()

    def toSyntaxTree(self):
        """Converts regular expression to an abstract syntax tree. Returns the tree."""
        self.addImpliedConcat()  # recognize concat
        self.operand_stack = []
        self.operator_stack = []

        for sym in self.regexp:  # sym in re
            if self.isOperand(sym):
                self.operand_stack.append(BinTree(sym))  # make tree with single node
            elif sym == "(":
                self.operator_stack.append("(")

            elif self.isOperator(
                sym
            ):  # if operator at the top of stack has higher precedence than scanned operator
                while (
                    self.operand_stack
                    and self.operator_stack
                    and self.getOperatorPrec(sym)
                    <= self.getOperatorPrec(
                        self.operator_stack[len(self.operator_stack) - 1]
                    )
                ):
                    self.popAndPushTree()  # tree pushed on operand stack

                self.operator_stack.append(sym)

            elif sym == ")":
                while self.operator_stack and (
                    self.operator_stack[len(self.operator_stack) - 1] != "("
                ):
                    self.popAndPushTree()  # tree pushed on operand stack

                if self.operator_stack:
                    scan_left_paren = self.operator_stack.pop()
                else:
                    raise InvalidExpression()
            else:
                raise InvalidExpression()

        while self.operator_stack:
            self.popAndPushTree()
        if self.operand_stack:
            abstrac_syntax_tree = (
                self.operand_stack.pop()
            )  # abs syntax tree at the top of the stack
        else:
            raise InvalidExpression()

        return abstrac_syntax_tree

    def syntaxTreeToNFA(self, syntax_tree):
        """Converts abstract syntax tree to an equivalent NFA. Returns the NFA"""
        if self.regexp == "e":  # check base case
            sub_tree_nfa = NFA()
            sub_tree_nfa = sub_tree_nfa.generateBaseEmpStr(self.alpha)
            return sub_tree_nfa

        if self.regexp == "N":  # check base case
            sub_tree_nfa = NFA()
            sub_tree_nfa = sub_tree_nfa.generateBaseEmpSet(self.alpha)
            return sub_tree_nfa

        stack = []
        for node in syntax_tree:  # pre order traversal of syntax tree
            val_node = node.getVal()
            if self.isOperand(val_node):  # base case NFA for leaf node
                sub_tree_nfa = NFA()
                sub_tree_nfa = sub_tree_nfa.generateBaseNFA(val_node, self.alpha)
                stack.append(sub_tree_nfa)
            elif self.isOperator(val_node):
                if val_node == "*":  # unary operator
                    if stack:
                        operand = stack.pop()
                    else:
                        raise InvalidExpression()
                    sub_tree_nfa = NFA()
                    sub_tree_nfa = sub_tree_nfa.generateStarNFA(
                        operand, self.alpha
                    )  # NFA for star operation
                    stack.append(sub_tree_nfa)
                else:  # binary operator
                    if stack:
                        operand_two = stack.pop()
                    else:
                        raise InvalidExpression()
                    if stack:
                        operand_one = stack.pop()
                    else:
                        raise InvalidExpression()
                    if val_node == "|":
                        sub_tree_nfa = NFA()
                        sub_tree_nfa = sub_tree_nfa.generateUnionNFA(
                            operand_one, operand_two, self.alpha
                        )  # NFA for union operation
                        stack.append(sub_tree_nfa)
                    elif val_node == " ":
                        sub_tree_nfa = NFA()
                        sub_tree_nfa = sub_tree_nfa.generateConcatNFA(
                            operand_one, operand_two, self.alpha
                        )  # NFA for conatenation operation
                        stack.append(sub_tree_nfa)
                    else:
                        raise InvalidExpression()
            else:
                raise InvalidExpression()

        # conversion stack should only have the equivalent NFA at the top of the stack
        if not stack:
            raise InvalidExpression()
        if len(stack) != 1:
            raise InvalidExpression()

        return stack.pop()  # return NFA


class BinTree:
    """Abstract Data Type used for syntax tree"""

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __iter__(self):
        return self.depth_first_search(self)

    def depth_first_search(self, tree):
        if tree.left:
            yield from tree.left
        if tree.right:
            yield from tree.right
        yield tree

    def getVal(self):
        return self.val

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right


class DFA:
    """Simulates a DFA"""

    def __init__(
        self, num_states=0, alpha="", transitions=[], start_state=0, accept_states=[]
    ):
        self.num_states = num_states
        self.alpha = alpha
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

        # before simulation
        self.next_state = None
        self.final_state = None

    # getter/setters
    def getNumStates(self):
        return self.num_states

    def setNumStates(self, num_states):
        self.num_states = num_states

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, alpha):
        self.alpha = alpha

    def getTransitions(self):
        return self.transitions

    def setTransitionns(self, transitions):
        for trans in transitions:
            trans[0] = str(trans[0])
            trans[2] = str(trans[2])
        self.transitions = transitions

    def getStartState(self):
        return self.start_state

    def setStartState(self, start_state):
        self.start_state = start_state

    def getAcceptStates(self):
        return self.accept_states

    def setAcceptStates(self, accept_states):
        self.accept_states = accept_states

    # simulation
    def simulate(self, str):
        """
        Simulates the DFA on input str.  Returns
        True if str is in the language of the DFA,
        and False if not.
        """
        # start of simulation
        self.num_transitions = 0

        # empty string case
        if str == "" and self.start_state in self.accept_states:
            return True

        for sym in str:
            if sym not in self.alpha:
                return False
            else:
                # pass start state initially to transition func
                if self.num_transitions == 0:
                    self.transition(self.start_state, sym)
                # pass next state to transition func
                else:
                    self.transition(self.next_state, sym)

        # after all symbols are read set final state
        self.final_state = self.next_state

        # accepting final state
        if self.final_state in self.accept_states:
            return True
        else:
            # rejecting final state
            return False

    def transition(self, cur_state, sym):
        # increment transition
        self.num_transitions += 1
        for trans in self.transitions:
            # current state matches with current symbol
            if (trans[0] == cur_state) and (trans[1] == sym):
                self.next_state = trans[2]  # go to next state


class NFA:
    """Converts NFA to DFA"""

    def __init__(
        self, num_states=0, alpha="", transitions=[], start_state=0, accept_states=[]
    ):
        self.num_states = num_states
        self.alpha = alpha
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    # getters/setters
    def getNumStates(self):
        return self.num_states

    def setNumStates(self, num_states):
        self.num_states = num_states

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, alpha):
        self.alpha = alpha

    def getTransitions(self):
        return self.transitions

    def setTransitionns(self, transitions):
        for trans in transitions:
            trans[0] = str(trans[0])
            trans[2] = str(trans[2])
        self.transitions = transitions

    def getStartState(self):
        return self.start_state

    def setStartState(self, start_state):
        self.start_state = start_state

    def getAcceptStates(self):
        return self.accept_states

    def setAcceptStates(self, accept_states):
        self.accept_states = accept_states

    # helper methods used to generate concat/union NFAS
    def reLabelStartState(self, num_states_shift):
        self.start_state += num_states_shift

    def reLabelAcceptStates(self, num_states_shift):
        relabeled_accept_states = []
        for state in self.accept_states:
            relabeled_accept_states.append(state + num_states_shift)
        self.accept_states = relabeled_accept_states

    def reLabelTransitions(self, num_states_shift):
        for trans in self.transitions:
            trans[0] = str(int(num_states_shift) + int(trans[0]))
            trans[2] = str(int(num_states_shift) + int(trans[2]))

    # base cases for NFA
    def generateBaseNFA(self, leaf, alpha):
        self.num_states_base = 2
        self.transitions_base = []
        self.transitions_base.append(["1", leaf, "2"])
        self.start_state_base = 1
        self.accept_states_base = []
        self.accept_states_base.append(2)

        baseNFA = NFA(
            self.num_states_base,
            alpha,
            self.transitions_base,
            self.start_state_base,
            self.accept_states_base,
        )

        return baseNFA

    def generateBaseEmpStr(self, alpha):
        return NFA(1, alpha, [], 1, [1])  # accept state

    def generateBaseEmpSet(self, alpha):
        return NFA(1, alpha, [], 1, [])  # no accept state

    def generateUnionNFA(self, left_nfa, right_nfa, alpha):
        # add new start state
        self.num_states_union = 1 + left_nfa.getNumStates() + right_nfa.getNumStates()
        self.start_state_union = self.num_states_union

        # relabel left NFA so we do not have the same state number labels in both NFAs
        left_nfa.reLabelStartState(right_nfa.getNumStates())
        left_nfa.reLabelTransitions(right_nfa.getNumStates())
        left_nfa.reLabelAcceptStates(right_nfa.getNumStates())

        # epsilons from new start state to left/right start states
        self.transitions_union = left_nfa.getTransitions() + right_nfa.getTransitions()
        self.transitions_union.append(
            [str(self.start_state_union), "e", str(left_nfa.getStartState())]
        )
        self.transitions_union.append(
            [str(self.start_state_union), "e", str(right_nfa.getStartState())]
        )

        # accept states of left and right NFA
        self.accept_states_union = (
            right_nfa.getAcceptStates() + left_nfa.getAcceptStates()
        )

        unionNFA = NFA(
            self.num_states_union,
            alpha,
            self.transitions_union,
            self.start_state_union,
            self.accept_states_union,
        )

        return unionNFA

    def generateConcatNFA(self, left_nfa, right_nfa, alpha):
        self.num_states_concat = left_nfa.getNumStates() + right_nfa.getNumStates()

        # relabel state nums
        right_nfa.reLabelStartState(left_nfa.getNumStates())
        right_nfa.reLabelTransitions(left_nfa.getNumStates())
        right_nfa.reLabelAcceptStates(left_nfa.getNumStates())

        self.start_state_concat = left_nfa.getStartState()

        self.transitions_concat = left_nfa.getTransitions() + right_nfa.getTransitions()

        # add epsilons from left nfa accept states to right nfa start state
        for state in left_nfa.getAcceptStates():
            self.transitions_concat.append(
                [str(state), "e", str(right_nfa.getStartState())]
            )

        # new accept states will be the right nfa accept states
        self.accept_states_concat = right_nfa.getAcceptStates()

        concatNFA = NFA(
            self.num_states_concat,
            alpha,
            self.transitions_concat,
            self.start_state_concat,
            self.accept_states_concat,
        )

        return concatNFA

    def generateStarNFA(self, single_nfa, alpha):
        # add new startstate
        self.num_states_star = 1 + single_nfa.getNumStates()
        self.start_state_star = self.num_states_star

        self.transitions_star = []
        self.transitions_star = single_nfa.getTransitions()

        # add epsilon transitions from new start state to old start state
        self.transitions_star.append(
            [str(self.start_state_star), "e", str(single_nfa.getStartState())]
        )

        # add epsilon transitions from old accept states to old start state
        for state in single_nfa.getAcceptStates():
            self.transitions_star.append(
                [str(state), "e", str(single_nfa.getStartState())]
            )

        self.accept_states_star = []
        self.accept_states_star.append(self.start_state_star)
        self.accept_states_star = single_nfa.getAcceptStates() + self.accept_states_star

        starNFA = NFA(
            self.num_states_star,
            alpha,
            self.transitions_star,
            self.start_state_star,
            self.accept_states_star,
        )

        return starNFA

    def generateEpsilonTransitions(self, states):
        """Returns frozen set of states that have epsilon transitions defined in the passed in states"""
        self.states_list_dfa = list(states)
        i = 0
        while i < len(self.states_list_dfa):
            for trans in self.transitions:  # scan trans func list for NFA
                if (
                    trans[0] == self.states_list_dfa[i]
                    and trans[1] == "e"
                    and trans[2] not in self.states_list_dfa
                ):  # look for e trans
                    self.states_list_dfa.append(trans[2])  # add to DFA state set
                # 	i = -1 #new state added to state list may also have e trans
                # break
            i += 1
        return frozenset(sorted(set(self.states_list_dfa)))

    def generateSymbolTransitions(self, symbol):
        """Returns set of next states that have valid symbol trans defined for all states in the current set of states"""
        self.cur_states_list_dfa = list(self.cur_state_set)

        for i in range(len(self.cur_states_list_dfa)):
            for trans in self.transitions:  # scan trans func list for NFA
                if (
                    trans[0] == self.cur_states_list_dfa[i]
                    and trans[1] == symbol
                    and trans[2] not in self.next_state_set
                ):  # check if new state needed
                    self.next_state_set.add(trans[2])  # add new state

        return set(self.next_state_set)

    def generateAcceptStates(self):
        """Void method to init accept states"""
        self.accept_states_dfa = set()
        for state in self.state_dict_dfa:
            if (
                len(state.intersection(self.accept_states)) > 0
            ):  # state in DFA set of states has at least one state in set of NFA accept states
                self.accept_states_dfa.add(
                    self.state_dict_dfa[state]
                )  # add DFA state to set of DFA accepting states

    def toDFA(self):
        """
        Converts the "self" NFA into an equivalent DFA
        and writes it to the file whose name is dfa_filename.
        The format of the DFA file must have the same format
        as described in the first programming assignment (pa1).
        This file must be able to be opened and simulated by your
        pa1 program.

        This function should not read in the NFA file again.  It should
        create the DFA from the internal representation of the NFA that you
        created in __init__.
        """

        # to string
        accept_states = []
        for state in self.accept_states:
            accept_states.append(str(state))
        self.accept_states = accept_states

        self.start_state = str(self.start_state)

        # DFA set of start states contains only the NFA start state
        start_state_set_dfa = set([self.start_state])
        # States reachable via epsilon transitions added to DFA set of start states
        start_state_set_dfa = self.generateEpsilonTransitions(
            start_state_set_dfa
        )  # ret hashable frozen set

        # {NFA state: DFA state} where NFA state is a immutable set and DFA state is an int
        self.state_dict_dfa = {}
        state_label_dfa = 1  # DFA state number
        self.state_dict_dfa[start_state_set_dfa] = state_label_dfa  # add start state

        # init cur state with no valid transitions for algoritim
        self.cur_state_set = start_state_set_dfa

        # keeps track of states without valid transitions defined for each sym in alpha
        stack = []
        stack.append(self.cur_state_set)

        self.transitions_dfa = []  # DFA trans func will be written in output file
        converted = False  # no DFA yet
        while (
            not converted or len(stack) > 0
        ):  # scan NFA states until DFA is fully constructed (valid transitions in each state for each sym in alpha)
            converted = True
            self.cur_state_set = set(
                stack.pop()
            )  # get DFA state without valid transition

            for (
                sym
            ) in (
                self.alpha
            ):  # define valid transition for each state for each symbol in alphabet
                self.next_state_set = set()  # no next state yet
                self.next_state_set = self.generateSymbolTransitions(sym)  # ret list
                self.next_state_set = self.generateEpsilonTransitions(
                    self.next_state_set
                )  # ret frozen set

                if (
                    self.next_state_set in self.state_dict_dfa
                ):  # if DFA state already exists
                    self.transitions_dfa.append(
                        [
                            self.state_dict_dfa[frozenset(self.cur_state_set)],
                            sym,
                            self.state_dict_dfa[frozenset(self.next_state_set)],
                        ]
                    )  # no need to create a new DFA state simply add valid DFA trans
                else:  # DFA state does not exist
                    state_label_dfa += 1  # add new state value
                    self.state_dict_dfa[frozenset(self.next_state_set)] = (
                        state_label_dfa  # new DFA state is constructed
                    )

                    self.transitions_dfa.append(
                        [
                            self.state_dict_dfa[frozenset(self.cur_state_set)],
                            sym,
                            self.state_dict_dfa[frozenset(self.next_state_set)],
                        ]
                    )  # add valid DFA trans
                    stack.append(
                        self.next_state_set
                    )  # new DFA state added to states that need valid transitions defined
                    converted = (
                        False  # created a new DFA w/o any valid transitions defined
                    )

        self.generateAcceptStates()
        self.num_states_dfa = len(self.state_dict_dfa)
        self.start_state_dfa = 1

        return DFA(
            self.num_states_dfa,
            self.alpha,
            self.transitions_dfa,
            self.start_state_dfa,
            self.accept_states_dfa,
        )
