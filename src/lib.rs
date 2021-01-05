use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

use safecast::{CastFrom, CastInto};
use serde::de::{self, SeqAccess, Visitor};
use serde::ser::Serializer;
use serde::{Deserialize, Deserializer, Serialize};

pub mod class;
pub mod instance;

pub use class::*;
pub use instance::*;
use serde::export::Formatter;

type _Complex<T> = num::complex::Complex<T>;

#[derive(Clone, Copy, Eq)]
pub enum Number {
    Bool(Boolean),
    Complex(Complex),
    Float(Float),
    Int(Int),
    UInt(UInt),
}

impl NumberInstance for Number {
    type Abs = Number;
    type Exp = Self;
    type Class = NumberType;

    fn class(&self) -> NumberType {
        match self {
            Self::Bool(_) => NumberType::Bool,
            Self::Complex(c) => c.class().into(),
            Self::Float(f) => f.class().into(),
            Self::Int(i) => i.class().into(),
            Self::UInt(u) => u.class().into(),
        }
    }

    fn into_type(self, dtype: NumberType) -> Number {
        use NumberType as NT;

        match dtype {
            NT::Bool => {
                let b: Boolean = self.cast_into();
                b.into()
            }
            NT::Complex(ct) => {
                let c: Complex = self.cast_into();
                c.into_type(ct).into()
            }
            NT::Float(ft) => {
                let f: Float = self.cast_into();
                f.into_type(ft).into()
            }
            NT::Int(it) => {
                let i: Int = self.cast_into();
                i.into_type(it).into()
            }
            NT::UInt(ut) => {
                let u: UInt = self.cast_into();
                u.into_type(ut).into()
            }
            NT::Number => self,
        }
    }

    fn abs(self) -> Number {
        use Number::*;
        match self {
            Complex(c) => Float(c.abs()),
            Float(f) => Float(f.abs()),
            Int(i) => Int(i.abs()),
            other => other,
        }
    }

    fn pow(self, exp: Self) -> Self {
        match (self, exp) {
            (Self::Complex(this), Self::Complex(exp)) => this.pow(exp).into(),
            (Self::Float(this), Self::Float(exp)) => this.pow(exp).into(),
            (Self::Int(this), Self::UInt(exp)) => this.pow(exp).into(),
            (Self::Int(this), Self::Int(that)) => {
                // pow(Int, -Int) doesn't make sense, so cast to Float
                Float::cast_from(this).pow(Float::cast_from(that)).into()
            }
            (Self::UInt(this), Self::UInt(exp)) => this.pow(exp).into(),
            (Self::Bool(this), Self::Bool(exp)) => this.pow(exp).into(),
            (this, exp) => {
                let dtype = Ord::max(self.class(), exp.class());
                let this = this.into_type(dtype);
                let exp = exp.into_type(dtype);
                this.pow(exp)
            }
        }
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(l), Self::Int(r)) => l.eq(r),
            (Self::UInt(l), Self::UInt(r)) => l.eq(r),
            (Self::Float(l), Self::Float(r)) => l.eq(r),
            (Self::Bool(l), Self::Bool(r)) => l.eq(r),
            (Self::Complex(l), Self::Complex(r)) => l.eq(r),

            (Self::Complex(l), r) => l.eq(&Complex::cast_from(*r)),
            (Self::Float(l), r) => l.eq(&Float::cast_from(*r)),
            (Self::Int(l), r) => l.eq(&Int::cast_from(*r)),
            (Self::UInt(l), r) => l.eq(&UInt::cast_from(*r)),

            (l, r) => r.eq(l),
        }
    }
}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Number) -> Option<Ordering> {
        match (self, other) {
            (Self::Int(l), Self::Int(r)) => l.partial_cmp(r),
            (Self::UInt(l), Self::UInt(r)) => l.partial_cmp(r),
            (Self::Float(l), Self::Float(r)) => l.partial_cmp(r),
            (Self::Bool(l), Self::Bool(r)) => l.partial_cmp(r),
            (Self::Complex(l), Self::Complex(r)) => l.partial_cmp(r),

            (Self::Complex(l), r) => l.partial_cmp(&Complex::cast_from(*r)),
            (Self::Float(l), r) => l.partial_cmp(&Float::cast_from(*r)),
            (Self::Int(l), r) => l.partial_cmp(&Int::cast_from(*r)),
            (Self::UInt(l), r) => l.partial_cmp(&UInt::cast_from(*r)),

            (l, r) => match r.partial_cmp(l) {
                Some(ordering) => Some(match ordering {
                    Ordering::Less => Ordering::Greater,
                    Ordering::Equal => Ordering::Equal,
                    Ordering::Greater => Ordering::Less,
                }),
                None => None,
            },
        }
    }
}

impl Add for Number {
    type Output = Self;

    fn add(self, other: Number) -> Self {
        let dtype = Ord::max(self.class(), other.class());

        use NumberType as NT;

        match dtype {
            NT::Bool => {
                let this: Boolean = self.cast_into();
                (this + other.cast_into()).into()
            }
            NT::Complex(_) => {
                let this: Complex = self.cast_into();
                (this + other.cast_into()).into()
            }
            NT::Float(_) => {
                let this: Float = self.cast_into();
                (this + other.cast_into()).into()
            }
            NT::Int(_) => {
                let this: Int = self.cast_into();
                (this + other.cast_into()).into()
            }
            NT::UInt(_) => {
                let this: UInt = self.cast_into();
                (this + other.cast_into()).into()
            }
            NT::Number => panic!("A number instance must have a specific type, not Number"),
        }
    }
}

impl Sub for Number {
    type Output = Self;

    fn sub(self, other: Number) -> Self {
        let dtype = Ord::max(self.class(), other.class());

        use NumberType as NT;

        match dtype {
            NT::Bool => {
                let this: Boolean = self.cast_into();
                (this - other.cast_into()).into()
            }
            NT::Complex(_) => {
                let this: Complex = self.cast_into();
                (this - other.cast_into()).into()
            }
            NT::Float(_) => {
                let this: Float = self.cast_into();
                (this - other.cast_into()).into()
            }
            NT::Int(_) => {
                let this: Int = self.cast_into();
                (this - other.cast_into()).into()
            }
            NT::UInt(_) => {
                let this: UInt = self.cast_into();
                (this - other.cast_into()).into()
            }
            NT::Number => panic!("A number instance must have a specific type, not Number"),
        }
    }
}

impl Mul for Number {
    type Output = Self;

    fn mul(self, other: Number) -> Self {
        let dtype = Ord::max(self.class(), other.class());

        use NumberType as NT;

        match dtype {
            NT::Bool => {
                let this: Boolean = self.cast_into();
                (this * other.cast_into()).into()
            }
            NT::Complex(_) => {
                let this: Complex = self.cast_into();
                (this * other.cast_into()).into()
            }
            NT::Float(_) => {
                let this: Float = self.cast_into();
                (this * other.cast_into()).into()
            }
            NT::Int(_) => {
                let this: Int = self.cast_into();
                (this * other.cast_into()).into()
            }
            NT::UInt(_) => {
                let this: UInt = self.cast_into();
                (this * other.cast_into()).into()
            }
            NT::Number => panic!("A number instance must have a specific type, not Number"),
        }
    }
}

impl Div for Number {
    type Output = Self;

    fn div(self, other: Number) -> Self {
        let dtype = Ord::max(self.class(), other.class());

        use NumberType as NT;

        match dtype {
            NT::Number => panic!("A number instance must have a specific type, not Number"),
            NT::Complex(_) => {
                let this: Complex = self.cast_into();
                (this / other.cast_into()).into()
            }
            NT::Float(_) => {
                let this: Float = self.cast_into();
                (this / other.cast_into()).into()
            }
            NT::Int(_) => {
                let this: Int = self.cast_into();
                (this / other.cast_into()).into()
            }
            NT::UInt(_) => {
                let this: UInt = self.cast_into();
                (this / other.cast_into()).into()
            }
            NT::Bool => {
                let this: Boolean = self.cast_into();
                (this / other.cast_into()).into()
            }
        }
    }
}

impl Default for Number {
    fn default() -> Self {
        Self::Bool(Boolean::default())
    }
}

impl From<bool> for Number {
    fn from(b: bool) -> Self {
        Self::Bool(b.into())
    }
}

impl From<Boolean> for Number {
    fn from(b: Boolean) -> Number {
        Number::Bool(b)
    }
}

impl From<u8> for Number {
    fn from(u: u8) -> Self {
        Self::UInt(u.into())
    }
}

impl From<u16> for Number {
    fn from(u: u16) -> Self {
        Self::UInt(u.into())
    }
}

impl From<u32> for Number {
    fn from(u: u32) -> Self {
        Self::UInt(u.into())
    }
}

impl From<u64> for Number {
    fn from(u: u64) -> Self {
        Self::UInt(u.into())
    }
}

impl From<UInt> for Number {
    fn from(u: UInt) -> Number {
        Number::UInt(u)
    }
}

impl From<i16> for Number {
    fn from(i: i16) -> Self {
        Self::Int(i.into())
    }
}

impl From<i32> for Number {
    fn from(i: i32) -> Self {
        Self::Int(i.into())
    }
}

impl From<i64> for Number {
    fn from(i: i64) -> Self {
        Self::Int(i.into())
    }
}

impl From<f32> for Number {
    fn from(f: f32) -> Self {
        Self::Float(f.into())
    }
}

impl From<f64> for Number {
    fn from(f: f64) -> Self {
        Self::Float(f.into())
    }
}

impl From<Float> for Number {
    fn from(f: Float) -> Number {
        Number::Float(f)
    }
}

impl From<Int> for Number {
    fn from(i: Int) -> Number {
        Number::Int(i)
    }
}

impl From<_Complex<f32>> for Number {
    fn from(c: _Complex<f32>) -> Self {
        Self::Complex(c.into())
    }
}

impl From<_Complex<f64>> for Number {
    fn from(c: _Complex<f64>) -> Self {
        Self::Complex(c.into())
    }
}

impl From<Complex> for Number {
    fn from(c: Complex) -> Number {
        Number::Complex(c)
    }
}

impl CastFrom<Number> for Boolean {
    fn cast_from(number: Number) -> Boolean {
        if number == number.class().zero() {
            false.into()
        } else {
            true.into()
        }
    }
}

impl CastFrom<Number> for Float {
    fn cast_from(number: Number) -> Float {
        use Number::*;
        match number {
            Bool(b) => Self::cast_from(b),
            Complex(c) => Self::cast_from(c),
            Float(f) => f,
            Int(i) => Self::cast_from(i),
            UInt(u) => Self::cast_from(u),
        }
    }
}

impl CastFrom<Number> for Int {
    fn cast_from(number: Number) -> Int {
        use Number::*;
        match number {
            Bool(b) => Self::cast_from(b),
            Complex(c) => Self::cast_from(c),
            Float(f) => Self::cast_from(f),
            Int(i) => i,
            UInt(u) => Self::cast_from(u),
        }
    }
}

impl CastFrom<Number> for UInt {
    fn cast_from(number: Number) -> UInt {
        use Number::*;
        match number {
            Bool(b) => Self::cast_from(b),
            Complex(c) => Self::cast_from(c),
            Float(f) => Self::cast_from(f),
            Int(i) => Self::cast_from(i),
            UInt(u) => u,
        }
    }
}

impl CastFrom<Number> for bool {
    fn cast_from(n: Number) -> bool {
        Boolean::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for _Complex<f32> {
    fn cast_from(n: Number) -> _Complex<f32> {
        Complex::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for _Complex<f64> {
    fn cast_from(n: Number) -> _Complex<f64> {
        Complex::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for f32 {
    fn cast_from(n: Number) -> f32 {
        Float::cast_from(n).cast_into()
    }
}
impl CastFrom<Number> for f64 {
    fn cast_from(n: Number) -> f64 {
        Float::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for i16 {
    fn cast_from(n: Number) -> i16 {
        Int::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for i32 {
    fn cast_from(n: Number) -> i32 {
        Int::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for i64 {
    fn cast_from(n: Number) -> i64 {
        Int::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for u8 {
    fn cast_from(n: Number) -> u8 {
        UInt::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for u16 {
    fn cast_from(n: Number) -> u16 {
        UInt::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for u32 {
    fn cast_from(n: Number) -> u32 {
        UInt::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for u64 {
    fn cast_from(n: Number) -> u64 {
        UInt::cast_from(n).cast_into()
    }
}

impl CastFrom<Number> for usize {
    fn cast_from(n: Number) -> usize {
        UInt::cast_from(n).cast_into()
    }
}

pub struct NumberVisitor;

impl<'de> Visitor<'de> for NumberVisitor {
    type Value = Number;

    fn expecting(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "a Number, like 1 or -2 or 3.14 or [0., -1.414]")
    }

    fn visit_bool<E: de::Error>(self, b: bool) -> Result<Self::Value, E> {
        Ok(Number::Bool(b.into()))
    }

    fn visit_i8<E: de::Error>(self, i: i8) -> Result<Self::Value, E> {
        Ok(Number::Int(Int::I16(i as i16)))
    }

    fn visit_i16<E: de::Error>(self, i: i16) -> Result<Self::Value, E> {
        Ok(Number::Int(Int::I16(i)))
    }

    fn visit_i32<E: de::Error>(self, i: i32) -> Result<Self::Value, E> {
        Ok(Number::Int(Int::I32(i)))
    }

    fn visit_i64<E: de::Error>(self, i: i64) -> Result<Self::Value, E> {
        Ok(Number::Int(Int::I64(i)))
    }

    fn visit_u8<E: de::Error>(self, u: u8) -> Result<Self::Value, E> {
        Ok(Number::UInt(UInt::U8(u)))
    }

    fn visit_u16<E: de::Error>(self, u: u16) -> Result<Self::Value, E> {
        Ok(Number::UInt(UInt::U16(u)))
    }

    fn visit_u32<E: de::Error>(self, u: u32) -> Result<Self::Value, E> {
        Ok(Number::UInt(UInt::U32(u)))
    }

    fn visit_u64<E>(self, u: u64) -> Result<Self::Value, E> {
        Ok(Number::UInt(UInt::U64(u)))
    }

    fn visit_f32<E>(self, f: f32) -> Result<Self::Value, E> {
        Ok(Number::Float(Float::F32(f)))
    }

    fn visit_f64<E>(self, f: f64) -> Result<Self::Value, E> {
        Ok(Number::Float(Float::F64(f)))
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> Result<Self::Value, <A as SeqAccess<'de>>::Error> {
        let re = seq
            .next_element()?
            .ok_or_else(|| de::Error::custom("Complex number missing real component"))?;

        let im = seq
            .next_element()?
            .ok_or_else(|| de::Error::custom("Complex number missing imaginary component"))?;

        Ok(Number::Complex(Complex::C64(_Complex::<f64>::new(re, im))))
    }
}

impl<'de> Deserialize<'de> for Number {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, <D as Deserializer<'de>>::Error> {
        deserializer.deserialize_any(NumberVisitor)
    }
}

impl Serialize for Number {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Number::Bool(b) => b.serialize(s),
            Number::Complex(c) => c.serialize(s),
            Number::Float(f) => f.serialize(s),
            Number::Int(i) => i.serialize(s),
            Number::UInt(u) => u.serialize(s),
        }
    }
}

impl fmt::Debug for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Number::Bool(b) => fmt::Display::fmt(b, f),
            Number::Complex(c) => fmt::Display::fmt(c, f),
            Number::Float(n) => fmt::Display::fmt(n, f),
            Number::Int(i) => fmt::Display::fmt(i, f),
            Number::UInt(u) => fmt::Display::fmt(u, f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let ones = [
            Number::from(true),
            Number::from(1u8),
            Number::from(1u16),
            Number::from(1u32),
            Number::from(1u64),
            Number::from(1i16),
            Number::from(1i32),
            Number::from(1i64),
            Number::from(1f32),
            Number::from(1f64),
            Number::from(_Complex::new(1f32, 0f32)),
            Number::from(_Complex::new(1f64, 0f64)),
        ];

        let f = Number::from(false);
        let t = Number::from(true);
        let two = Number::from(2);

        for one in &ones {
            let one = *one;
            let zero = one.class().zero();

            assert_eq!(one, one.class().one());

            assert_eq!(two, one * two);
            assert_eq!(one, (one * two) - one);
            assert_eq!(two, (one * two) / one);
            assert_eq!(zero, one * zero);

            assert_eq!(one, one.pow(zero));
            assert_eq!(one * one, one.pow(two));
            assert_eq!(two.pow(two), (one * two).pow(two));

            assert_eq!(f, one.not());
            assert_eq!(f, one.and(zero));
            assert_eq!(t, one.or(zero));
            assert_eq!(t, one.xor(zero));
        }
    }

    #[test]
    fn test_serialize() {
        let numbers = [
            Number::from(false),
            Number::from(12u16),
            Number::from(-3),
            Number::from(3.14),
            Number::from(_Complex::<f32>::new(0., -1.414)),
        ];

        for number in &numbers {
            let compare: Number =
                serde_json::from_str(&serde_json::to_string(number).unwrap()).unwrap();
            assert_eq!(number, &compare);
        }
    }
}
