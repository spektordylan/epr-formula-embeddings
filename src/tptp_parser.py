"""
Minimal TPTP FOF parser + EPR (Bernays-Schoenfinkel) validator.

Covers the subset of the TPTP FOF language needed for EPR problems:
quantifiers (! / ?), connectives (~ & | => <=>), predicates and
0-arity constants. Deliberately does NOT support function symbols
of positive arity, arithmetic, or CNF/TFF/THF syntax - if a formula
uses a function symbol, we flag it as non-EPR rather than trying to
parse arbitrary FOF.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Union


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass
class Var:
    name: str


@dataclass
class Term:
    """A predicate/function application. arity 0 => constant."""
    name: str
    args: List[Union["Term", Var]] = field(default_factory=list)


@dataclass
class Not:
    formula: "Formula"


@dataclass
class BinOp:
    op: str  # '&', '|', '=>', '<=>'
    left: "Formula"
    right: "Formula"


@dataclass
class Quant:
    kind: str  # '!' or '?'
    vars: List[str]
    formula: "Formula"


@dataclass
class Atom:
    """A predicate application used as an atomic formula, or an equality."""
    term: Term


Formula = Union[Not, BinOp, Quant, Atom]


@dataclass
class FofUnit:
    name: str
    role: str
    formula: Formula


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"""
    \s*(?:
        (?P<comment>%[^\n]*)
    |   (?P<quoted>'(?:[^'\\]|\\.)*')
    |   (?P<iff><=>)
    |   (?P<implies>=>)
    |   (?P<neq>!=)
    |   (?P<punct>[()\[\],.:])
    |   (?P<op>[!?&|~=])
    |   (?P<dollar>\$[A-Za-z0-9_]+)
    |   (?P<ident>[A-Za-z0-9_]+)
    )
""", re.VERBOSE)


def tokenize(text: str) -> List[str]:
    tokens = []
    pos = 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            if text[pos].isspace():
                pos += 1
                continue
            raise SyntaxError(f"Unexpected character {text[pos]!r} at {pos}")
        pos = m.end()
        if m.lastgroup == "comment":
            continue
        value = m.group(m.lastgroup)
        if m.lastgroup == "quoted":
            # Treat a 'quoted identifier' as a single opaque name token,
            # stripping the surrounding quotes.
            value = value[1:-1]
        tokens.append(value)
    return tokens


# ---------------------------------------------------------------------------
# Parser (recursive descent)
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[str]):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def expect(self, tok):
        t = self.next()
        if t != tok:
            raise ParseError(f"Expected {tok!r}, got {t!r} at token {self.i}")
        return t

    # fof(name, role, formula).
    def parse_units(self) -> List[FofUnit]:
        units = []
        while self.peek() is not None:
            units.append(self.parse_unit())
        return units

    def parse_unit(self) -> FofUnit:
        kind = self.peek()
        if kind == "fof":
            return self._parse_fof_unit()
        if kind == "cnf":
            return self._parse_cnf_unit()
        raise ParseError(f"Expected 'fof' or 'cnf', got {kind!r} at token {self.i}")

    def _parse_fof_unit(self) -> FofUnit:
        self.expect("fof")
        self.expect("(")
        name = self.next()
        self.expect(",")
        role = self.next()
        self.expect(",")
        formula = self.parse_formula()
        self.expect(")")
        self.expect(".")
        return FofUnit(name=name, role=role, formula=formula)

    def _parse_cnf_unit(self) -> FofUnit:
        # cnf(name, role, clause).
        # clause = literal | ( literal (| literal)* )
        # every variable in a clause is implicitly universally quantified;
        # there are no explicit quantifiers or nested connectives besides
        # negation and disjunction.
        self.expect("cnf")
        self.expect("(")
        name = self.next()
        self.expect(",")
        role = self.next()
        self.expect(",")
        formula = self.parse_clause()
        self.expect(")")
        self.expect(".")
        return FofUnit(name=name, role=role, formula=formula)

    def parse_clause(self) -> Formula:
        if self.peek() == "(":
            self.next()
            lit = self.parse_literal()
            while self.peek() == "|":
                self.next()
                lit = BinOp("|", lit, self.parse_literal())
            self.expect(")")
            return lit
        return self.parse_literal()

    def parse_literal(self) -> Formula:
        negated = False
        if self.peek() == "~":
            self.next()
            negated = True
        formula = self.parse_atom()  # may itself be Not(Atom(...)) for !=
        if negated:
            # collapse ~(~x) -> x instead of double-wrapping
            if isinstance(formula, Not):
                return formula.formula
            return Not(formula)
        return formula

    # Precedence (loosest to tightest): <=>  =>  |  &  ~  quantifier  atom
    def parse_formula(self) -> Formula:
        return self.parse_iff()

    def parse_iff(self) -> Formula:
        left = self.parse_implies()
        if self.peek() == "<=>":
            self.next()
            right = self.parse_implies()
            return BinOp("<=>", left, right)
        return left

    def parse_implies(self) -> Formula:
        left = self.parse_or()
        if self.peek() == "=>":
            self.next()
            right = self.parse_implies()  # right-assoc
            return BinOp("=>", left, right)
        return left

    def parse_or(self) -> Formula:
        left = self.parse_and()
        while self.peek() == "|":
            self.next()
            right = self.parse_and()
            left = BinOp("|", left, right)
        return left

    def parse_and(self) -> Formula:
        left = self.parse_unary()
        while self.peek() == "&":
            self.next()
            right = self.parse_unary()
            left = BinOp("&", left, right)
        return left

    def parse_unary(self) -> Formula:
        if self.peek() == "~":
            self.next()
            return Not(self.parse_unary())
        if self.peek() in ("!", "?"):
            return self.parse_quant()
        return self.parse_primary()

    def parse_quant(self) -> Formula:
        kind = self.next()  # ! or ?
        self.expect("[")
        varlist = [self.next()]
        while self.peek() == ",":
            self.next()
            varlist.append(self.next())
        self.expect("]")
        self.expect(":")
        body = self.parse_unary()
        return Quant(kind, varlist, body)

    def parse_primary(self) -> Formula:
        if self.peek() == "(":
            self.next()
            f = self.parse_formula()
            self.expect(")")
            return f
        return self.parse_atom()

    def parse_atom(self) -> Formula:
        """Parses `pred(args)` or `term = term` / `term != term`.
        Returns Atom, or Not(Atom) for a `!=` (disequality)."""
        lhs = self.parse_arg()
        if self.peek() in ("=", "!="):
            eq_op = self.next()
            rhs = self.parse_arg()
            atom = Atom(Term("=", [lhs, rhs]))
            return Not(atom) if eq_op == "!=" else atom
        return Atom(lhs)

    def parse_term(self) -> Term:
        name = self.next()
        args = []
        if self.peek() == "(":
            self.next()
            args.append(self.parse_arg())
            while self.peek() == ",":
                self.next()
                args.append(self.parse_arg())
            self.expect(")")
        return Term(name, args)

    def parse_arg(self) -> Union[Term, Var]:
        tok = self.peek()
        # Convention: uppercase-leading identifiers are variables
        if tok[0].isupper():
            self.next()
            return Var(tok)
        return self.parse_term()


def parse_fof_file(text: str) -> List[FofUnit]:
    return Parser(tokenize(text)).parse_units()


# ---------------------------------------------------------------------------
# EPR validation
# ---------------------------------------------------------------------------

def term_uses_function_symbol(t: Union[Term, Var]) -> bool:
    """True if t (as a *sub*-term, not top-level atom) is anything other
    than a variable or a 0-arity constant - i.e. a real function symbol."""
    if isinstance(t, Var):
        return False
    if isinstance(t, Term):
        if len(t.args) > 0:
            return True
        return False
    return False


def formula_uses_function_symbol(f: Formula) -> bool:
    if isinstance(f, Atom):
        # top-level term is the predicate application; check its ARGUMENTS
        # for nested function symbols (arguments must be vars/constants)
        return any(term_uses_function_symbol(a) for a in f.term.args)
    if isinstance(f, Not):
        return formula_uses_function_symbol(f.formula)
    if isinstance(f, BinOp):
        return (formula_uses_function_symbol(f.left)
                or formula_uses_function_symbol(f.right))
    if isinstance(f, Quant):
        return formula_uses_function_symbol(f.formula)
    raise TypeError(f"Unknown formula node {f!r}")


def is_epr_eligible(units: List[FofUnit]) -> Tuple[bool, str]:
    """Checks the syntactic condition that actually defines EPR-eligibility:
    no function symbols of positive arity anywhere (relation symbols and
    0-arity constants only). Quantifier order in the source file doesn't
    have to already be prenex ∃*∀* - that's a normal form reachable by
    prenexing + Skolemization, which preserves this no-function-symbols
    property iff it held before Skolemization for a formula with only
    existential quantifiers ahead of the universals in the relevant scope.
    For a first-pass filter, the no-function-symbols check is the useful,
    cheaply computable one.
    """
    for u in units:
        if formula_uses_function_symbol(u.formula):
            return False, f"formula '{u.name}' contains a function symbol"
    return True, "no function symbols found"


if __name__ == "__main__":
    sample = """
%--------------------------------------------------------------------------
% Status   : Unsatisfiable
% SPC      : FOF_UNS_EPR
%--------------------------------------------------------------------------
fof(ax1, axiom, ! [X] : (p(X) => ? [Y] : (q(X,Y) & r(Y))) ).
fof(ax2, axiom, ! [X,Y] : (r(Y) => ~q(X,Y)) ).
fof(conj, conjecture, ~ (? [X] : p(X)) ).
"""
    units = parse_fof_file(sample)
    for u in units:
        print(f"{u.name} ({u.role}): {u.formula}")
    ok, reason = is_epr_eligible(units)
    print(f"\nEPR-eligible: {ok} ({reason})")