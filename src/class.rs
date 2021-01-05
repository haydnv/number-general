use std::cmp::Ordering;
use std::fmt;
use std::iter::{Product, Sum};
use std::ops::{Add, Div, Mul, Sub};

use safecast::*;
use serde::{Deserialize, Serialize};

use super::instance::{Boolean, Complex, Float, Int, UInt};
use super::{Number, _Complex};

/// Defines common properties of numeric types supported by [`Number`].
pub trait NumberClass: Into<NumberType> + Ord + Send + fmt::Display {
    type Instance: NumberInstance;

    fn size(self) -> usize;

    fn one(&self) -> <Self as NumberClass>::Instance;

    fn zero(&self) -> <Self as NumberClass>::Instance;
}

/// Defines common operations on numeric types supported by [`Number`].
pub trait NumberInstance:
    Copy
    + Default
    + Sized
    + PartialOrd
    + From<Boolean>
    + Into<Number>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Product
    + Sum
    + fmt::Debug
    + fmt::Display
{
    type Abs: NumberInstance;
    type Exp: NumberInstance;
    type Class: NumberClass<Instance = Self>;

    /// Get an impl of [`NumberClass`] describing this number.
    fn class(&self) -> Self::Class;

    /// Cast this number into the specified [`NumberClass`].
    fn into_type(
        self,
        dtype: <Self as NumberInstance>::Class,
    ) -> <<Self as NumberInstance>::Class as NumberClass>::Instance;

    /// Calculate the absolute value of this number.
    fn abs(self) -> Self::Abs;

    /// Raise this number to the given exponent.
    fn pow(self, exp: Self::Exp) -> Self;

    /// Return `true` if `self` and `other` are nonzero.
    fn and(self, other: Self) -> Self
    where
        Boolean: CastFrom<Self>,
    {
        Boolean::cast_from(self)
            .and(Boolean::cast_from(other))
            .into()
    }

    /// Return `true` if this number is zero.
    fn not(self) -> Self
    where
        Boolean: CastFrom<Self>,
    {
        Boolean::cast_from(self).not().into()
    }

    /// Return `true` if `self` or `other` is nonzero.
    fn or(self, other: Self) -> Self
    where
        Boolean: CastFrom<Self>,
    {
        let this = Boolean::cast_from(self);
        let that = Boolean::cast_from(other);
        this.or(that).into()
    }

    /// Return `true` if exactly one of `self` and `other` is zero.
    fn xor(self, other: Self) -> Self
    where
        Self: CastInto<Boolean>,
    {
        let zero = self.class().zero();

        if self != zero && other == zero {
            self
        } else if self == zero && other != zero {
            other
        } else {
            zero
        }
    }
}

/// The type of a [`Complex`] number.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum ComplexType {
    C32,
    C64,
    Complex,
}

impl NumberClass for ComplexType {
    type Instance = Complex;

    fn size(self) -> usize {
        match self {
            Self::C32 => 8,
            Self::C64 => 16,
            Self::Complex => 16,
        }
    }

    fn one(&self) -> Complex {
        match self {
            Self::C32 => _Complex::<f32>::new(1f32, 0f32).into(),
            Self::C64 => _Complex::<f64>::new(1f64, 0f64).into(),
            Self::Complex => _Complex::<f32>::new(1f32, 0f32).into(),
        }
    }

    fn zero(&self) -> Complex {
        match self {
            Self::C32 => _Complex::<f32>::new(0f32, 0f32).into(),
            Self::C64 => _Complex::<f64>::new(0f64, 0f64).into(),
            Self::Complex => _Complex::<f32>::new(0f32, 0f32).into(),
        }
    }
}

impl Ord for ComplexType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::C32, Self::C32) => Ordering::Equal,
            (Self::C64, Self::C64) => Ordering::Equal,
            (Self::Complex, Self::Complex) => Ordering::Equal,

            (Self::Complex, _) => Ordering::Greater,
            (_, Self::Complex) => Ordering::Less,

            (Self::C64, Self::C32) => Ordering::Greater,
            (Self::C32, Self::C64) => Ordering::Less,
        }
    }
}

impl PartialOrd for ComplexType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<ComplexType> for NumberType {
    fn from(ct: ComplexType) -> NumberType {
        Self::Complex(ct)
    }
}

impl fmt::Debug for ComplexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ComplexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::C32 => write!(f, "32-bit complex number"),
            Self::C64 => write!(f, "64-bit complex number"),
            Self::Complex => write!(f, "complex number"),
        }
    }
}

/// The type of a [`Boolean`].
#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct BooleanType;

impl NumberClass for BooleanType {
    type Instance = Boolean;

    fn size(self) -> usize {
        1
    }

    fn one(&self) -> Boolean {
        true.into()
    }

    fn zero(&self) -> Boolean {
        false.into()
    }
}

impl Ord for BooleanType {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl PartialOrd for BooleanType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<BooleanType> for NumberType {
    fn from(_bt: BooleanType) -> NumberType {
        NumberType::Bool
    }
}

impl fmt::Debug for BooleanType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for BooleanType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Boolean")
    }
}

/// The type of a [`Float`].
#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum FloatType {
    F32,
    F64,
    Float,
}

impl NumberClass for FloatType {
    type Instance = Float;

    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::Float => 8,
        }
    }

    fn one(&self) -> Float {
        match self {
            Self::F32 => 1f32.into(),
            Self::F64 => 1f64.into(),
            Self::Float => 1f32.into(),
        }
    }

    fn zero(&self) -> Float {
        match self {
            Self::F32 => 0f32.into(),
            Self::F64 => 0f64.into(),
            Self::Float => 0f32.into(),
        }
    }
}

impl Ord for FloatType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::F32, Self::F32) => Ordering::Equal,
            (Self::F64, Self::F64) => Ordering::Equal,
            (Self::Float, Self::Float) => Ordering::Equal,

            (Self::Float, _) => Ordering::Greater,
            (_, Self::Float) => Ordering::Less,

            (Self::F64, Self::F32) => Ordering::Greater,
            (Self::F32, Self::F64) => Ordering::Less,
        }
    }
}

impl PartialOrd for FloatType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<FloatType> for NumberType {
    fn from(ft: FloatType) -> NumberType {
        NumberType::Float(ft)
    }
}

impl fmt::Debug for FloatType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for FloatType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use FloatType::*;
        match self {
            F32 => write!(f, "32-bit float"),
            F64 => write!(f, "64-bit float"),
            Float => write!(f, "float"),
        }
    }
}

/// The type of an [`Int`].
#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum IntType {
    I16,
    I32,
    I64,
    Int,
}

impl NumberClass for IntType {
    type Instance = Int;

    fn size(self) -> usize {
        match self {
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Int => 8,
        }
    }

    fn one(&self) -> Int {
        match self {
            Self::I16 => 1i16.into(),
            Self::I32 => 1i32.into(),
            Self::I64 => 1i64.into(),
            Self::Int => 1i16.into(),
        }
    }

    fn zero(&self) -> Int {
        match self {
            Self::I16 => 0i16.into(),
            Self::I32 => 0i32.into(),
            Self::I64 => 0i64.into(),
            Self::Int => 0i16.into(),
        }
    }
}

impl Ord for IntType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::I16, Self::I16) => Ordering::Equal,
            (Self::I32, Self::I32) => Ordering::Equal,
            (Self::I64, Self::I64) => Ordering::Equal,
            (Self::Int, Self::Int) => Ordering::Equal,

            (Self::Int, _) => Ordering::Greater,
            (_, Self::Int) => Ordering::Less,

            (Self::I64, _) => Ordering::Greater,
            (_, Self::I64) => Ordering::Less,

            (Self::I16, _) => Ordering::Less,
            (_, Self::I16) => Ordering::Greater,
        }
    }
}

impl PartialOrd for IntType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<IntType> for NumberType {
    fn from(it: IntType) -> NumberType {
        NumberType::Int(it)
    }
}

impl fmt::Debug for IntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for IntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::I16 => write!(f, "16-bit integer"),
            Self::I32 => write!(f, "32-bit integer"),
            Self::I64 => write!(f, "64-bit integer"),
            Self::Int => write!(f, "integer"),
        }
    }
}

/// The type of a [`UInt`].
#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum UIntType {
    U8,
    U16,
    U32,
    U64,
    UInt,
}

impl NumberClass for UIntType {
    type Instance = UInt;

    fn size(self) -> usize {
        match self {
            UIntType::U8 => 1,
            UIntType::U16 => 2,
            UIntType::U32 => 4,
            UIntType::U64 => 8,
            UIntType::UInt => 8,
        }
    }

    fn one(&self) -> UInt {
        match self {
            Self::U8 => 1u8.into(),
            Self::U16 => 1u16.into(),
            Self::U32 => 1u32.into(),
            Self::U64 => 1u64.into(),
            Self::UInt => 1u8.into(),
        }
    }

    fn zero(&self) -> UInt {
        match self {
            Self::U8 => 0u8.into(),
            Self::U16 => 0u16.into(),
            Self::U32 => 0u32.into(),
            Self::U64 => 0u64.into(),
            Self::UInt => 0u8.into(),
        }
    }
}

impl Ord for UIntType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::U8, Self::U8) => Ordering::Equal,
            (Self::U16, Self::U16) => Ordering::Equal,
            (Self::U32, Self::U32) => Ordering::Equal,
            (Self::U64, Self::U64) => Ordering::Equal,
            (Self::UInt, Self::UInt) => Ordering::Equal,

            (Self::UInt, _) => Ordering::Greater,
            (_, Self::UInt) => Ordering::Less,

            (Self::U64, _) => Ordering::Greater,
            (_, Self::U64) => Ordering::Less,

            (Self::U8, _) => Ordering::Less,
            (_, Self::U8) => Ordering::Greater,

            (Self::U32, Self::U16) => Ordering::Greater,
            (Self::U16, Self::U32) => Ordering::Less,
        }
    }
}

impl PartialOrd for UIntType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<UIntType> for NumberType {
    fn from(ut: UIntType) -> NumberType {
        NumberType::UInt(ut)
    }
}

impl fmt::Debug for UIntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for UIntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use UIntType::*;
        match self {
            U8 => write!(f, "8-bit unsigned"),
            U16 => write!(f, "16-bit unsigned"),
            U32 => write!(f, "32-bit unsigned"),
            U64 => write!(f, "64-bit unsigned"),
            UInt => write!(f, "uint"),
        }
    }
}

/// The type of a generic [`Number`].
#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum NumberType {
    Bool,
    Complex(ComplexType),
    Float(FloatType),
    Int(IntType),
    UInt(UIntType),
    Number,
}

impl NumberType {
    pub fn uint64() -> Self {
        NumberType::UInt(UIntType::U64)
    }
}

impl NumberClass for NumberType {
    type Instance = Number;

    fn size(self) -> usize {
        use NumberType::*;
        match self {
            Bool => 1,
            Complex(ct) => NumberClass::size(ct),
            Float(ft) => NumberClass::size(ft),
            Int(it) => NumberClass::size(it),
            UInt(ut) => NumberClass::size(ut),

            // a generic Number still has a distinct maximum size
            Number => NumberClass::size(ComplexType::C64),
        }
    }

    fn one(&self) -> Number {
        use NumberType::*;
        match self {
            Bool | Number => true.into(),
            Complex(ct) => ct.one().into(),
            Float(ft) => ft.one().into(),
            Int(it) => it.one().into(),
            UInt(ut) => ut.one().into(),
        }
    }

    fn zero(&self) -> Number {
        use NumberType::*;
        match self {
            Bool | Number => false.into(),
            Complex(ct) => ct.zero().into(),
            Float(ft) => ft.zero().into(),
            Int(it) => it.zero().into(),
            UInt(ut) => ut.zero().into(),
        }
    }
}

impl Ord for NumberType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Bool, Self::Bool) => Ordering::Equal,
            (Self::Complex(l), Self::Complex(r)) => l.cmp(r),
            (Self::Float(l), Self::Float(r)) => l.cmp(r),
            (Self::Int(l), Self::Int(r)) => l.cmp(r),
            (Self::UInt(l), Self::UInt(r)) => l.cmp(r),

            (Self::Number, Self::Number) => Ordering::Equal,
            (Self::Number, _) => Ordering::Greater,
            (_, Self::Number) => Ordering::Less,

            (Self::Bool, _) => Ordering::Less,
            (_, Self::Bool) => Ordering::Greater,

            (Self::Complex(_), _) => Ordering::Greater,
            (_, Self::Complex(_)) => Ordering::Less,

            (Self::UInt(_), Self::Int(_)) => Ordering::Less,
            (Self::UInt(_), Self::Float(_)) => Ordering::Less,
            (Self::Int(_), Self::UInt(_)) => Ordering::Greater,
            (Self::Float(_), Self::UInt(_)) => Ordering::Greater,
            (Self::Int(_), Self::Float(_)) => Ordering::Less,
            (Self::Float(_), Self::Int(_)) => Ordering::Greater,
        }
    }
}

impl PartialOrd for NumberType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for NumberType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for NumberType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use NumberType::*;
        match self {
            Bool => fmt::Debug::fmt(&BooleanType, f),
            Complex(ct) => fmt::Debug::fmt(ct, f),
            Float(ft) => fmt::Debug::fmt(ft, f),
            Int(it) => fmt::Debug::fmt(it, f),
            UInt(ut) => fmt::Debug::fmt(ut, f),
            Number => write!(f, "Number"),
        }
    }
}
