#![allow(private_bounds)]
// Super minimal symbolic algebra library

use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{Add, BitAnd, BitOr, Div, IndexMut, Mul, Rem, Sub},
};

use tinyvec::ArrayVec;

/// A symbolic expression stored on the stack
pub type Expression = GenericExpression<ArrayVec<[Term; 20]>>; // We need to figure out how to reduce this, can't be fixed at 20. ShapeTracker would take up 6 dims * 12 pads * 12 slices * 20 terms * 8 bytes = 138kb
/// A symbolic expression stored on the heap
pub type BigExpression = GenericExpression<Vec<Term>>;

/// Trait implemented on the 2 main symbolic expression storage types, Vec<Term> and ArrayVec<Term>
#[allow(clippy::len_without_is_empty)]
trait ExpressionStorage:
    IndexMut<usize, Output = Term> + std::iter::Extend<Term> + IntoIterator<Item = Term> + Default
{
    fn len(&self) -> usize;
    fn push(&mut self, term: Term);
    fn pop(&mut self) -> Option<Term>;
    fn remove(&mut self, index: usize) -> Term;
}

// Implement the main storage types
impl ExpressionStorage for Vec<Term> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn push(&mut self, term: Term) {
        Vec::push(self, term)
    }
    fn pop(&mut self) -> Option<Term> {
        Vec::pop(self)
    }
    fn remove(&mut self, index: usize) -> Term {
        Vec::remove(self, index)
    }
}

impl<const C: usize> ExpressionStorage for ArrayVec<[Term; C]>
where
    [Term; C]: tinyvec::Array<Item = Term>,
{
    fn len(&self) -> usize {
        ArrayVec::len(self)
    }
    fn push(&mut self, term: Term) {
        ArrayVec::push(self, term)
    }
    fn pop(&mut self) -> Option<Term> {
        ArrayVec::pop(self)
    }
    fn remove(&mut self, index: usize) -> Term {
        ArrayVec::remove(self, index)
    }
}

/// A symbolic expression
#[derive(Clone, Default)]
pub struct GenericExpression<S: ExpressionStorage> {
    pub terms: S,
}

impl<S: Copy + ExpressionStorage> Copy for GenericExpression<S> {}

impl<S: PartialEq + ExpressionStorage> PartialEq for GenericExpression<S> {
    fn eq(&self, other: &Self) -> bool {
        self.terms == other.terms
    }
}

impl<S: ExpressionStorage + Clone> Debug for GenericExpression<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut symbols = vec![];
        for term in self.terms.clone() {
            let new_symbol = match term {
                Term::Num(n) => n.to_string(),
                Term::Var(c) => c.to_string(),
                Term::Max => format!(
                    "max({}, {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Min => format!(
                    "min({}, {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                _ => format!(
                    "({}{term:?}{})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
            };
            symbols.push(new_symbol);
        }
        write!(f, "{}", symbols.pop().unwrap())
    }
}

impl<S: ExpressionStorage> GenericExpression<S> {
    /// Minimise the expression down to its smallest form
    pub fn minimize(mut self) -> Self {
        let mut i = 0;
        while i < self.terms.len().saturating_sub(2) {
            match (self.terms[i], self.terms[i + 1], self.terms[i + 2]) {
                (Term::Num(b), Term::Num(a), term) if term.as_op().is_some() => {
                    self.terms[i] = Term::Num(term.as_op().unwrap()(a, b));
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove min(i, inf) and min(inf, i)
                (Term::Num(b), Term::Num(_) | Term::Var(_), Term::Min)
                    if b == i32::MAX as usize =>
                {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                (_, Term::Num(a), Term::Min) if a == i32::MAX as usize => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove max(i, 0) and max(0, i)
                (Term::Num(0), Term::Num(_) | Term::Var(_), Term::Max) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                (_, Term::Num(0), Term::Max) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove i + 0, i - 0 and 0 + i
                (_, Term::Num(0), Term::Add) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                (Term::Num(0), Term::Num(_) | Term::Var(_), Term::Add | Term::Sub) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Remove i * 0, 0 * i
                (Term::Num(0), Term::Num(_) | Term::Var(_), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                (Term::Num(_) | Term::Var(_), Term::Num(0), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Remove 0 / i
                (Term::Num(_) | Term::Var(_), Term::Num(0), Term::Div) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Remove i * 1 and 1 * i
                (Term::Num(1), Term::Num(_) | Term::Var(_), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                (_, Term::Num(1), Term::Mul) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Remove i / 1
                (Term::Num(1), Term::Num(_) | Term::Var(_), Term::Div) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Simplify i - i to 0
                (Term::Var(b), Term::Var(a), Term::Sub) if a == b => {
                    self.terms[i] = Term::Num(0);
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Simplify true && i to i
                (_, Term::Num(1), Term::And) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Simplify i && true to i
                (Term::Num(1), _, Term::And) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Simplify false || i to i
                (_, Term::Num(0), Term::Or) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Simplify i || false to i
                (Term::Num(0), _, Term::Or) => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i);
                }
                // Simplify i >= i to true (1)
                (Term::Var(b), Term::Var(a), Term::Gte) if a == b => {
                    self.terms[i] = Term::Num(1);
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Simplify i < i to false (0)
                (Term::Var(b), Term::Var(a), Term::Lt) if a == b => {
                    self.terms[i] = Term::Num(0);
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Simplify min(i, i) to i
                (Term::Var(b), Term::Var(a), Term::Min) if a == b => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                // Simplify max(i, i) to i
                (Term::Var(b), Term::Var(a), Term::Max) if a == b => {
                    self.terms.remove(i + 2);
                    self.terms.remove(i + 1);
                }
                _ => {
                    i += 1;
                }
            }
        }

        self
    }

    pub fn min<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Min);
        rhs.minimize()
    }

    pub fn max<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Max);
        rhs.minimize()
    }

    pub fn gte<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Gte);
        rhs.minimize()
    }

    pub fn lt<E: Into<Self>>(self, rhs: E) -> Self {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Lt);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage + Clone> GenericExpression<S> {
    /// Evaluate the expression with no variables. Returns Some(value) if no variables are required, otherwise returns None.
    pub fn to_usize(&self) -> Option<usize> {
        self.exec(&HashMap::default())
    }
    /// Evaluate the expression with one value for all variables.
    pub fn exec_single_var(&self, value: usize) -> usize {
        let mut stack = Vec::new();
        for term in self.terms.clone() {
            match term {
                Term::Num(n) => stack.push(n),
                Term::Var(_) => stack.push(value),
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b));
                }
            }
        }
        stack.pop().unwrap()
    }
    /// Evaluate the expression given variables.
    pub fn exec(&self, variables: &HashMap<char, usize>) -> Option<usize> {
        let mut stack = Vec::new();
        for term in self.terms.clone() {
            match term {
                Term::Num(n) => stack.push(n),
                Term::Var(c) => {
                    if let Some(n) = variables.get(&c) {
                        stack.push(*n)
                    } else {
                        return None;
                    }
                }
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b));
                }
            }
        }
        stack.pop()
    }
    /// Retrieve all symbols in the expression.
    pub fn to_symbols(&self) -> Vec<char> {
        self.terms
            .clone()
            .into_iter()
            .filter_map(|t| match t {
                Term::Var(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    /// Check if the '-' variable exists in the expression.
    pub fn is_unknown(&self) -> bool {
        self.terms
            .clone()
            .into_iter()
            .any(|t| matches!(t, Term::Var('-')))
    }
}

/// A single term of a symbolic expression such as a variable, number or operation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Term {
    Num(usize),
    Var(char),
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    And,
    Or,
    Gte,
    Lt,
}

impl Debug for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Num(n) => write!(f, "{n}"),
            Term::Var(c) => write!(f, "{c}"),
            Term::Add => write!(f, "+"),
            Term::Sub => write!(f, "-"),
            Term::Mul => write!(f, "*"),
            Term::Div => write!(f, "/"),
            Term::Mod => write!(f, "%"),
            Term::Min => write!(f, "min"),
            Term::Max => write!(f, "max"),
            Term::And => write!(f, "&&"),
            Term::Or => write!(f, "||"),
            Term::Gte => write!(f, ">="),
            Term::Lt => write!(f, "<"),
        }
    }
}

impl Default for Term {
    fn default() -> Self {
        Self::Num(0)
    }
}

impl Term {
    pub fn as_op(self) -> Option<fn(usize, usize) -> usize> {
        match self {
            Term::Add => Some(std::ops::Add::add),
            Term::Sub => Some(|a, b| a.saturating_sub(b)),
            Term::Mul => Some(std::ops::Mul::mul),
            Term::Div => Some(std::ops::Div::div),
            Term::Mod => Some(std::ops::Rem::rem),
            Term::Max => Some(core::cmp::Ord::max),
            Term::Min => Some(core::cmp::Ord::min),
            Term::And => Some(|a, b| (a != 0 && b != 0) as usize),
            Term::Or => Some(|a, b| (a != 0 || b != 0) as usize),
            Term::Gte => Some(|a, b| (a >= b) as usize),
            Term::Lt => Some(|a, b| (a < b) as usize),
            _ => None,
        }
    }
}

impl<S: ExpressionStorage> From<Term> for GenericExpression<S> {
    fn from(value: Term) -> Self {
        let mut terms = S::default();
        terms.push(value);
        GenericExpression { terms }
    }
}

impl<S: ExpressionStorage> From<char> for GenericExpression<S> {
    fn from(value: char) -> Self {
        GenericExpression::from(Term::Var(value))
    }
}

impl<S: ExpressionStorage> From<&char> for GenericExpression<S> {
    fn from(value: &char) -> Self {
        GenericExpression::from(Term::Var(*value))
    }
}

impl<S: ExpressionStorage> From<usize> for GenericExpression<S> {
    fn from(value: usize) -> Self {
        GenericExpression::from(Term::Num(value))
    }
}

impl<S: ExpressionStorage> From<&usize> for GenericExpression<S> {
    fn from(value: &usize) -> Self {
        GenericExpression::from(Term::Num(*value))
    }
}

impl<S: ExpressionStorage> From<i32> for GenericExpression<S> {
    fn from(value: i32) -> Self {
        GenericExpression::from(value as usize)
    }
}

impl<S: ExpressionStorage> From<&i32> for GenericExpression<S> {
    fn from(value: &i32) -> Self {
        GenericExpression::from(*value as usize)
    }
}

impl From<Expression> for BigExpression {
    fn from(value: Expression) -> Self {
        Self {
            terms: value.terms.to_vec(),
        }
    }
}

impl From<BigExpression> for Expression {
    fn from(value: BigExpression) -> Self {
        let mut terms = ArrayVec::new();
        terms.extend(value.terms);
        Self { terms }
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Add<E> for GenericExpression<S> {
    type Output = Self;
    fn add(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Add);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Sub<E> for GenericExpression<S> {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Sub);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Mul<E> for GenericExpression<S> {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mul);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Div<E> for GenericExpression<S> {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Div);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> Rem<E> for GenericExpression<S> {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Mod);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitAnd<E> for GenericExpression<S> {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::And);
        rhs.minimize()
    }
}

impl<S: ExpressionStorage, E: Into<Self>> BitOr<E> for GenericExpression<S> {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let mut rhs = rhs.into();
        rhs.terms.extend(self.terms);
        rhs.terms.push(Term::Or);
        rhs.minimize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_expressions() {
        let n = (Expression::from('x') + Term::Num(255)) / Term::Num(256) * Term::Num(256);
        assert_eq!(n.exec(&HashMap::from([('x', 767)])).unwrap(), 768);

        let n = (Expression::from('x') + Term::Num(255)) / Term::Num(256) * Term::Num(256);
        assert_eq!(n.exec(&HashMap::from([('x', 767)])).unwrap(), 768);
    }
}
