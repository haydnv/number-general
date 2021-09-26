use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::{Product, Sum};
use std::ops::*;
use std::str::FromStr;

use async_trait::async_trait;
use collate::Collate;
use destream::{Decoder, EncodeSeq, Encoder, FromStream, IntoStream, ToStream};
use futures::TryFutureExt;
use num::traits::Pow;
use safecast::*;
use serde::ser::{Serialize, SerializeSeq, Serializer};

use super::class::*;
use super::{Error, Number, _Complex};
use serde::{Deserialize, Deserializer};

const ERR_COMPLEX_POWER: &str = "complex exponent is not yet supported";

/// A boolean value.
#[derive(Clone, Copy, Hash, Ord, PartialEq, PartialOrd)]
pub struct Boolean(bool);

impl NumberInstance for Boolean {
    type Abs = Self;
    type Exp = Float;
    type Class = BooleanType;

    fn class(&self) -> BooleanType {
        BooleanType
    }

    fn into_type(self, _dtype: BooleanType) -> Boolean {
        self
    }

    fn abs(self) -> Self {
        self
    }

    fn exp(self) -> Self::Exp {
        Float::cast_from(self).exp()
    }

    fn pow(self, exp: Number) -> Self {
        if exp.cast_into() {
            self
        } else {
            self.class().one()
        }
    }

    fn and(self, other: Self) -> Self {
        Self(self.0 && other.0)
    }

    fn not(self) -> Self {
        Self(!self.0)
    }

    fn or(self, other: Self) -> Self {
        Self(self.0 || other.0)
    }

    fn xor(self, other: Self) -> Self {
        Self(self.0 ^ other.0)
    }
}

impl Default for Boolean {
    fn default() -> Boolean {
        Self(false)
    }
}

impl From<bool> for Boolean {
    fn from(b: bool) -> Boolean {
        Self(b)
    }
}

impl FromStr for Boolean {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        bool::from_str(s).map(Self::from).map_err(Error::new)
    }
}

impl From<Boolean> for bool {
    fn from(b: Boolean) -> bool {
        b.0
    }
}

impl From<&Boolean> for bool {
    fn from(b: &Boolean) -> bool {
        b.0
    }
}

impl Eq for Boolean {}

impl Add for Boolean {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        self.or(other)
    }
}

impl AddAssign for Boolean {
    fn add_assign(&mut self, other: Self) {
        self.0 |= other.0
    }
}

impl Sub for Boolean {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.xor(other)
    }
}

impl SubAssign for Boolean {
    fn sub_assign(&mut self, other: Self) {
        self.0 ^= other.0
    }
}

impl Sum for Boolean {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        Self(iter.any(|b| b.0))
    }
}

impl Mul for Boolean {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self.and(other)
    }
}

impl MulAssign for Boolean {
    fn mul_assign(&mut self, other: Self) {
        self.0 &= other.0
    }
}

impl Div for Boolean {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        if let Self(false) = other {
            panic!("divide by zero!")
        } else {
            self
        }
    }
}

impl DivAssign for Boolean {
    fn div_assign(&mut self, other: Self) {
        let div = *self / other;
        *self = div;
    }
}

impl Product for Boolean {
    fn product<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        Self(iter.all(|b| b.0))
    }
}

impl CastFrom<Boolean> for u64 {
    fn cast_from(b: Boolean) -> u64 {
        UInt::from(b).into()
    }
}

impl fmt::Debug for Boolean {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.class(), self)
    }
}

impl fmt::Display for Boolean {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Serialize for Boolean {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bool(self.0)
    }
}

impl<'de> Deserialize<'de> for Boolean {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        bool::deserialize(deserializer).map(Self::from)
    }
}

#[async_trait]
impl FromStream for Boolean {
    type Context = ();

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        bool::from_stream(cxt, decoder).map_ok(Self::from).await
    }
}

impl<'en> ToStream<'en> for Boolean {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        e.encode_bool(self.0)
    }
}

impl<'en> IntoStream<'en> for Boolean {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        e.encode_bool(self.0)
    }
}

/// A complex number.
#[derive(Clone, Copy)]
pub enum Complex {
    C32(_Complex<f32>),
    C64(_Complex<f64>),
}

impl NumberInstance for Complex {
    type Abs = Float;
    type Exp = Self;
    type Class = ComplexType;

    fn class(&self) -> ComplexType {
        match self {
            Self::C32(_) => ComplexType::C32,
            Self::C64(_) => ComplexType::C64,
        }
    }

    fn into_type(self, dtype: ComplexType) -> Complex {
        use ComplexType::*;
        match dtype {
            C32 => match self {
                Self::C64(c) => Self::C32(_Complex::new(c.re as f32, c.im as f32)),
                this => this,
            },
            C64 => match self {
                Self::C32(c) => Self::C64(_Complex::new(c.re as f64, c.im as f64)),
                this => this,
            },
            Complex => self,
        }
    }

    fn abs(self) -> Float {
        match self {
            Self::C32(c) => Float::F32(c.norm_sqr()),
            Self::C64(c) => Float::F64(c.norm_sqr()),
        }
    }

    fn exp(self) -> Self::Exp {
        match self {
            Self::C32(c) => Self::C32(c.exp()),
            Self::C64(c) => Self::C64(c.exp()),
        }
    }

    fn pow(self, exp: Number) -> Self {
        match self {
            Self::C64(this) => match exp {
                Number::Complex(exp) => unimplemented!("{}: {}", ERR_COMPLEX_POWER, exp),
                Number::Float(Float::F64(exp)) => Self::C64(this.pow(exp)),
                Number::Float(Float::F32(exp)) => Self::C64(this.pow(exp)),
                exp => Self::C64(this.pow(f64::cast_from(exp))),
            },
            Self::C32(this) => match exp {
                Number::Complex(exp) => unimplemented!("{}: {}", ERR_COMPLEX_POWER, exp),
                Number::Float(Float::F64(exp)) => {
                    let this = _Complex::<f64>::new(this.re.into(), this.im.into());
                    Self::C64(this.pow(exp))
                }
                Number::Float(Float::F32(exp)) => Self::C32(this.pow(exp)),
                exp => Self::C32(this.pow(f32::cast_from(exp))),
            },
        }
    }
}

impl FloatInstance for Complex {
    fn is_infinite(&self) -> bool {
        match self {
            Self::C32(c) => c.im.is_infinite() || c.re.is_infinite(),
            Self::C64(c) => c.im.is_infinite() || c.re.is_infinite(),
        }
    }

    fn is_nan(&self) -> bool {
        match self {
            Self::C32(c) => c.im.is_nan() || c.re.is_nan(),
            Self::C64(c) => c.im.is_nan() || c.re.is_nan(),
        }
    }
}

impl CastFrom<[f32; 2]> for Complex {
    fn cast_from(arr: [f32; 2]) -> Self {
        Self::C32(num::Complex::new(arr[0], arr[1]))
    }
}

impl CastFrom<[f64; 2]> for Complex {
    fn cast_from(arr: [f64; 2]) -> Self {
        Self::C64(num::Complex::new(arr[0], arr[1]))
    }
}

impl<R, I> CastFrom<(R, I)> for Complex
where
    R: CastInto<f64>,
    I: CastInto<f64>,
{
    fn cast_from(value: (R, I)) -> Self {
        let re = value.0.cast_into();
        let im = value.1.cast_into();
        Self::C64(num::Complex::new(re, im))
    }
}

impl CastFrom<Number> for Complex {
    fn cast_from(number: Number) -> Complex {
        use Number::*;
        match number {
            Number::Bool(b) => Self::from(b),
            Complex(c) => c,
            Float(f) => Self::from(f),
            Int(i) => Self::from(i),
            UInt(u) => Self::from(u),
        }
    }
}

impl CastFrom<Complex> for Boolean {
    fn cast_from(c: Complex) -> Self {
        match c {
            Complex::C32(c) if c.norm_sqr() == 0f32 => Self(false),
            Complex::C64(c) if c.norm_sqr() == 0f64 => Self(false),
            _ => Self(true),
        }
    }
}

impl CastFrom<Complex> for _Complex<f32> {
    fn cast_from(c: Complex) -> Self {
        match c {
            Complex::C32(c) => c,
            Complex::C64(_Complex { re, im }) => Self::new(re as f32, im as f32),
        }
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, other: Complex) -> Self {
        match (self, other) {
            (Self::C32(l), Self::C32(r)) => Self::C32(l + r),
            (Self::C64(l), Self::C64(r)) => Self::C64(l + r),
            (Self::C64(l), r) => {
                let r: _Complex<f64> = r.into();
                Self::C64(l + r)
            }
            (l, r) => r + l,
        }
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, other: Self) {
        let sum = *self + other;
        *self = sum;
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, other: Complex) -> Self {
        match (self, other) {
            (Self::C32(l), Self::C32(r)) => Self::C32(l - r),
            (l, r) => {
                let l: _Complex<f64> = l.into();
                let r: _Complex<f64> = r.into();
                Self::C64(l - r)
            }
        }
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, other: Self) {
        let diff = *self - other;
        *self = diff;
    }
}

impl Sum for Complex {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = ComplexType::Complex.zero();
        for i in iter {
            sum = sum + i;
        }
        sum
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, other: Complex) -> Self {
        match (self, other) {
            (Self::C32(l), Self::C32(r)) => Self::C32(l * r),
            (Self::C64(l), Self::C64(r)) => Self::C64(l * r),
            (Self::C64(l), r) => {
                let r: _Complex<f64> = r.into();
                Self::C64(l * r)
            }
            (l, r) => r * l,
        }
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, other: Self) {
        let product = *self * other;
        *self = product;
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, other: Complex) -> Self {
        match (self, other) {
            (Self::C32(l), Self::C32(r)) => Self::C32(l / r),
            (Self::C64(l), Self::C64(r)) => Self::C64(l / r),
            (Self::C64(l), r) => {
                let r: _Complex<f64> = r.into();
                Self::C64(l / r)
            }
            (Self::C32(l), Self::C64(r)) => {
                let l = _Complex::<f64>::new(l.re as f64, l.im as f64);
                Self::C64(l / r)
            }
        }
    }
}

impl DivAssign for Complex {
    fn div_assign(&mut self, other: Self) {
        let div = *self / other;
        *self = div;
    }
}

impl Product for Complex {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = ComplexType::Complex.zero();
        let mut product = ComplexType::Complex.one();

        for i in iter {
            if i == zero {
                return zero;
            }

            product = product * i;
        }
        product
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::C32(l), Self::C32(r)) => l.eq(r),
            (Self::C64(l), Self::C64(r)) => l.eq(r),
            (Self::C64(l), Self::C32(r)) => l.re as f32 == r.re && l.im as f32 == r.im,
            (l, r) => r.eq(l),
        }
    }
}

impl Eq for Complex {}

impl Hash for Complex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::C32(c) => {
                Float::F32(c.re).hash(state);
                Float::F32(c.im).hash(state);
            }
            Self::C64(c) => {
                Float::F64(c.re).hash(state);
                Float::F64(c.im).hash(state);
            }
        }
    }
}

impl Default for Complex {
    fn default() -> Complex {
        Complex::C32(_Complex::<f32>::default())
    }
}

impl From<Complex> for _Complex<f64> {
    fn from(c: Complex) -> Self {
        match c {
            Complex::C32(c) => Self::new(c.re as f64, c.im as f64),
            Complex::C64(c64) => c64,
        }
    }
}

impl From<Float> for Complex {
    fn from(f: Float) -> Self {
        match f {
            Float::F64(f) => Self::C64(_Complex::new(f, 0.0f64)),
            Float::F32(f) => Self::C32(_Complex::new(f, 0.0f32)),
        }
    }
}

impl From<Int> for Complex {
    fn from(i: Int) -> Self {
        match i {
            Int::I8(i) => Self::C32(_Complex::new(i as f32, 0.0f32)),
            Int::I64(i) => Self::C64(_Complex::new(i as f64, 0.0f64)),
            Int::I32(i) => Self::C32(_Complex::new(i as f32, 0.0f32)),
            Int::I16(i) => Self::C32(_Complex::new(i as f32, 0.0f32)),
        }
    }
}

impl From<UInt> for Complex {
    fn from(u: UInt) -> Self {
        match u {
            UInt::U64(u) => Self::C64(_Complex::new(u as f64, 0.0f64)),
            UInt::U32(u) => Self::C32(_Complex::new(u as f32, 0.0f32)),
            UInt::U16(u) => Self::C32(_Complex::new(u as f32, 0.0f32)),
            UInt::U8(u) => Self::C32(_Complex::new(u as f32, 0.0f32)),
        }
    }
}

impl From<Boolean> for Complex {
    fn from(b: Boolean) -> Self {
        match b {
            Boolean(true) => Self::C32(_Complex::new(1.0f32, 0.0f32)),
            Boolean(false) => Self::C32(_Complex::new(1.0f32, 0.0f32)),
        }
    }
}

impl From<_Complex<f32>> for Complex {
    fn from(c: _Complex<f32>) -> Self {
        Self::C32(c)
    }
}

impl From<_Complex<f64>> for Complex {
    fn from(c: _Complex<f64>) -> Self {
        Self::C64(c)
    }
}

impl FromStr for Complex {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        num::Complex::<f64>::from_str(s)
            .map(Self::from)
            .map_err(Error::new)
    }
}

impl<'de> Deserialize<'de> for Complex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let [re, im]: [f64; 2] = Deserialize::deserialize(deserializer)?;
        Ok(num::Complex::new(re, im).into())
    }
}

impl Serialize for Complex {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Complex::C32(c) => {
                let mut seq = s.serialize_seq(Some(2))?;
                seq.serialize_element(&c.re)?;
                seq.serialize_element(&c.im)?;
                seq.end()
            }
            Complex::C64(c) => {
                let mut seq = s.serialize_seq(Some(2))?;
                seq.serialize_element(&c.re)?;
                seq.serialize_element(&c.im)?;
                seq.end()
            }
        }
    }
}

#[async_trait]
impl FromStream for Complex {
    type Context = ();

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let [re, im]: [f64; 2] = FromStream::from_stream(cxt, decoder).await?;
        Ok(num::Complex::new(re, im).into())
    }
}

impl<'en> ToStream<'en> for Complex {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Complex::C32(c) => {
                let mut seq = e.encode_seq(Some(2))?;
                seq.encode_element(&c.re)?;
                seq.encode_element(&c.im)?;
                seq.end()
            }
            Complex::C64(c) => {
                let mut seq = e.encode_seq(Some(2))?;
                seq.encode_element(&c.re)?;
                seq.encode_element(&c.im)?;
                seq.end()
            }
        }
    }
}

impl<'en> IntoStream<'en> for Complex {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Complex::C32(c) => {
                let mut seq = e.encode_seq(Some(2))?;
                seq.encode_element(c.re)?;
                seq.encode_element(c.im)?;
                seq.end()
            }
            Complex::C64(c) => {
                let mut seq = e.encode_seq(Some(2))?;
                seq.encode_element(c.re)?;
                seq.encode_element(c.im)?;
                seq.end()
            }
        }
    }
}

impl fmt::Debug for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.class(), self)
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Complex::C32(c) => fmt::Display::fmt(c, f),
            Complex::C64(c) => fmt::Display::fmt(c, f),
        }
    }
}

/// Defines a collation order for [`Complex`].
#[derive(Default, Clone)]
pub struct ComplexCollator {
    float: FloatCollator,
}

impl Collate for ComplexCollator {
    type Value = Complex;

    fn compare(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        self.float.compare(&left.abs(), &right.abs())
    }
}

/// A floating-point number.
#[derive(Clone, Copy)]
pub enum Float {
    F32(f32),
    F64(f64),
}

impl NumberInstance for Float {
    type Abs = Float;
    type Exp = Self;
    type Class = FloatType;

    fn class(&self) -> FloatType {
        match self {
            Self::F32(_) => FloatType::F32,
            Self::F64(_) => FloatType::F64,
        }
    }

    fn into_type(self, dtype: FloatType) -> Float {
        use FloatType::*;
        match dtype {
            F32 => match self {
                Self::F64(f) => Self::F32(f as f32),
                this => this,
            },
            F64 => match self {
                Self::F32(f) => Self::F64(f as f64),
                this => this,
            },
            Float => self,
        }
    }

    fn abs(self) -> Float {
        match self {
            Self::F32(f) => Self::F32(f.abs()),
            Self::F64(f) => Self::F64(f.abs()),
        }
    }

    fn exp(self) -> Self::Exp {
        match self {
            Self::F32(f) => Self::F32(f.exp()),
            Self::F64(f) => Self::F64(f.exp()),
        }
    }

    fn pow(self, exp: Number) -> Self {
        match self {
            Self::F64(this) => match exp {
                Number::Complex(exp) => unimplemented!("{}: {}", ERR_COMPLEX_POWER, exp),
                exp => this.pow(f64::cast_from(exp)).into(),
            },
            Self::F32(this) => match exp {
                Number::Complex(exp) => unimplemented!("{}: {}", ERR_COMPLEX_POWER, exp),
                Number::Float(Float::F64(exp)) => Self::F64(f64::from(self).pow(exp)),
                exp => this.pow(f32::cast_from(exp)).into(),
            },
        }
    }
}

impl FloatInstance for Float {
    fn is_infinite(&self) -> bool {
        match self {
            Self::F32(f) => f.is_infinite(),
            Self::F64(f) => f.is_infinite(),
        }
    }

    fn is_nan(&self) -> bool {
        match self {
            Self::F32(f) => f.is_nan(),
            Self::F64(f) => f.is_nan(),
        }
    }
}

impl Eq for Float {}

impl Add for Float {
    type Output = Self;

    fn add(self, other: Float) -> Self {
        match (self, other) {
            (Self::F32(l), Self::F32(r)) => Self::F32(l + r),
            (Self::F64(l), Self::F64(r)) => Self::F64(l + r),
            (Self::F64(l), Self::F32(r)) => Self::F64(l + r as f64),
            (l, r) => (r + l),
        }
    }
}

impl AddAssign for Float {
    fn add_assign(&mut self, other: Self) {
        let sum = *self + other;
        *self = sum;
    }
}

impl Sub for Float {
    type Output = Self;

    fn sub(self, other: Float) -> Self {
        match (self, other) {
            (Self::F32(l), Self::F32(r)) => Self::F32(l - r),
            (l, r) => {
                let l: f64 = l.into();
                let r: f64 = r.into();
                Self::F64(l - r)
            }
        }
    }
}

impl SubAssign for Float {
    fn sub_assign(&mut self, other: Self) {
        let diff = *self - other;
        *self = diff;
    }
}

impl Sum for Float {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = FloatType::Float.zero();
        for i in iter {
            sum = sum + i;
        }
        sum
    }
}

impl Mul for Float {
    type Output = Self;

    fn mul(self, other: Float) -> Self {
        match (self, other) {
            (Self::F32(l), Self::F32(r)) => Self::F32(l * r),
            (Self::F64(l), Self::F64(r)) => Self::F64(l * r),
            (Self::F64(l), Self::F32(r)) => Self::F64(l * r as f64),
            (l, r) => (r * l),
        }
    }
}

impl MulAssign for Float {
    fn mul_assign(&mut self, other: Self) {
        let product = *self * other;
        *self = product;
    }
}

impl Div for Float {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        match (self, other) {
            (Self::F32(l), Self::F32(r)) => Self::F32(l / r),
            (Self::F64(l), Self::F64(r)) => Self::F64(l / r),
            (Self::F32(l), Self::F64(r)) => Self::F64((l as f64) / r),
            (Self::F64(l), Self::F32(r)) => Self::F64(l / (r as f64)),
        }
    }
}

impl DivAssign for Float {
    fn div_assign(&mut self, other: Self) {
        let div = *self / other;
        *self = div;
    }
}

impl Product for Float {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = FloatType::Float.zero();
        let mut product = FloatType::Float.one();

        for i in iter {
            if i == zero {
                return zero;
            }

            product = product * i;
        }
        product
    }
}

impl Hash for Float {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::F32(f) => f.to_be_bytes().hash(state),
            Self::F64(f) => f.to_be_bytes().hash(state),
        }
    }
}

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F32(l), Self::F32(r)) => l.eq(r),
            (Self::F64(l), Self::F64(r)) => l.eq(r),
            (Self::F64(l), Self::F32(r)) => (*l as f32).eq(r),
            (l, r) => r.eq(l),
        }
    }
}

impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Float) -> Option<Ordering> {
        match (self, other) {
            (Float::F32(l), Float::F32(r)) => l.partial_cmp(r),
            (Float::F64(l), Float::F64(r)) => l.partial_cmp(r),
            (l, r) => f64::from(*l).partial_cmp(&f64::from(*r)),
        }
    }
}

impl Default for Float {
    fn default() -> Float {
        Float::F32(f32::default())
    }
}

impl From<Boolean> for Float {
    fn from(b: Boolean) -> Self {
        match b {
            Boolean(true) => Self::F32(1.0f32),
            Boolean(false) => Self::F32(0.0f32),
        }
    }
}

impl From<f32> for Float {
    fn from(f: f32) -> Self {
        Self::F32(f)
    }
}

impl From<f64> for Float {
    fn from(f: f64) -> Self {
        Self::F64(f)
    }
}

impl From<Int> for Float {
    fn from(i: Int) -> Self {
        match i {
            Int::I8(i) => Self::F32(i as f32),
            Int::I64(i) => Self::F64(i as f64),
            Int::I32(i) => Self::F32(i as f32),
            Int::I16(i) => Self::F32(i as f32),
        }
    }
}

impl From<UInt> for Float {
    fn from(u: UInt) -> Self {
        match u {
            UInt::U64(u) => Self::F64(u as f64),
            UInt::U32(u) => Self::F32(u as f32),
            UInt::U16(u) => Self::F32(u as f32),
            UInt::U8(u) => Self::F32(u as f32),
        }
    }
}

impl FromStr for Float {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        f64::from_str(s).map(Self::from).map_err(Error::new)
    }
}

impl CastFrom<Complex> for Float {
    fn cast_from(c: Complex) -> Float {
        use Complex::*;
        match c {
            C32(c) => Self::F32(c.re),
            C64(c) => Self::F64(c.re),
        }
    }
}

impl CastFrom<Float> for Boolean {
    fn cast_from(f: Float) -> Boolean {
        use Float::*;
        let b = match f {
            F32(f) if f == 0f32 => false,
            F64(f) if f == 0f64 => false,
            _ => true,
        };

        Boolean(b)
    }
}

impl CastFrom<Float> for f32 {
    fn cast_from(f: Float) -> f32 {
        match f {
            Float::F32(f) => f,
            Float::F64(f) => f as f32,
        }
    }
}

impl From<Float> for f64 {
    fn from(f: Float) -> f64 {
        match f {
            Float::F32(f) => f as f64,
            Float::F64(f) => f,
        }
    }
}

impl<'de> Deserialize<'de> for Float {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        f64::deserialize(deserializer).map(Self::from)
    }
}

impl Serialize for Float {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Float::F32(f) => s.serialize_f32(*f),
            Float::F64(f) => s.serialize_f64(*f),
        }
    }
}

#[async_trait]
impl FromStream for Float {
    type Context = ();

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        f64::from_stream(cxt, decoder).map_ok(Self::from).await
    }
}

impl<'en> ToStream<'en> for Float {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match *self {
            Float::F32(f) => e.encode_f32(f),
            Float::F64(f) => e.encode_f64(f),
        }
    }
}

impl<'en> IntoStream<'en> for Float {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Float::F32(f) => e.encode_f32(f),
            Float::F64(f) => e.encode_f64(f),
        }
    }
}

impl fmt::Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.class(), self)
    }
}

impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Float::F32(n) => fmt::Display::fmt(n, f),
            Float::F64(n) => fmt::Display::fmt(n, f),
        }
    }
}

/// Defines a collation order for [`Float`].
#[derive(Default, Clone)]
pub struct FloatCollator;

impl Collate for FloatCollator {
    type Value = Float;

    fn compare(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        if let Some(order) = left.partial_cmp(right) {
            order
        } else {
            let left = f32::cast_from(*left);
            let right = f32::cast_from(*right);

            if left == right {
                Ordering::Equal
            } else if left == f32::NEG_INFINITY || right == f32::INFINITY {
                Ordering::Less
            } else if left == f32::INFINITY || right == f32::NEG_INFINITY {
                Ordering::Greater
            } else {
                panic!("no collation defined between {} and {}", left, right)
            }
        }
    }
}

/// A signed integer.
#[derive(Clone, Copy, Hash)]
pub enum Int {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
}

impl NumberInstance for Int {
    type Abs = Self;
    type Exp = Float;
    type Class = IntType;

    fn class(&self) -> IntType {
        match self {
            Self::I8(_) => IntType::I8,
            Self::I16(_) => IntType::I16,
            Self::I32(_) => IntType::I32,
            Self::I64(_) => IntType::I64,
        }
    }

    fn into_type(self, dtype: IntType) -> Int {
        use IntType::*;
        match dtype {
            I8 => match self {
                Self::I16(i) => Self::I8(i as i8),
                Self::I32(i) => Self::I8(i as i8),
                Self::I64(i) => Self::I8(i as i8),
                this => this,
            },
            I16 => match self {
                Self::I8(i) => Self::I16(i as i16),
                Self::I32(i) => Self::I16(i as i16),
                Self::I64(i) => Self::I16(i as i16),
                this => this,
            },
            I32 => match self {
                Self::I8(i) => Self::I32(i as i32),
                Self::I16(i) => Self::I32(i as i32),
                Self::I64(i) => Self::I32(i as i32),
                this => this,
            },
            I64 => match self {
                Self::I8(i) => Self::I64(i as i64),
                Self::I16(i) => Self::I64(i as i64),
                Self::I32(i) => Self::I64(i as i64),
                this => this,
            },
            Int => self,
        }
    }

    fn abs(self) -> Self {
        match self {
            Self::I8(i) => Int::I8(i.abs()),
            Self::I16(i) => Int::I16(i.abs()),
            Self::I32(i) => Int::I32(i.abs()),
            Self::I64(i) => Int::I64(i.abs()),
        }
    }

    fn exp(self) -> Self::Exp {
        Float::from(self).exp()
    }

    fn pow(self, exp: Number) -> Self {
        if exp < exp.class().zero() {
            return self.class().zero();
        }

        match self {
            Self::I8(this) => Self::I8(this.pow(exp.cast_into())),
            Self::I16(this) => Self::I16(this.pow(exp.cast_into())),
            Self::I32(this) => Self::I32(this.pow(exp.cast_into())),
            Self::I64(this) => Self::I64(this.pow(exp.cast_into())),
        }
    }
}

impl CastFrom<Complex> for Int {
    fn cast_from(c: Complex) -> Int {
        use Complex::*;
        match c {
            C32(c) => Self::I32(c.re as i32),
            C64(c) => Self::I64(c.re as i64),
        }
    }
}

impl CastFrom<Float> for Int {
    fn cast_from(f: Float) -> Int {
        use Float::*;
        match f {
            F32(f) => Self::I32(f as i32),
            F64(f) => Self::I64(f as i64),
        }
    }
}

impl CastFrom<Int> for Boolean {
    fn cast_from(i: Int) -> Boolean {
        use Int::*;
        let b = match i {
            I16(i) if i == 0i16 => false,
            I32(i) if i == 0i32 => false,
            I64(i) if i == 0i64 => false,
            _ => true,
        };

        Boolean(b)
    }
}

impl CastFrom<Int> for i16 {
    fn cast_from(i: Int) -> i16 {
        match i {
            Int::I8(i) => i as i16,
            Int::I16(i) => i,
            Int::I32(i) => i as i16,
            Int::I64(i) => i as i16,
        }
    }
}

impl CastFrom<Int> for i32 {
    fn cast_from(i: Int) -> i32 {
        match i {
            Int::I8(i) => i as i32,
            Int::I16(i) => i as i32,
            Int::I32(i) => i,
            Int::I64(i) => i as i32,
        }
    }
}

impl Eq for Int {}

impl Add for Int {
    type Output = Self;

    fn add(self, other: Int) -> Self {
        match (self, other) {
            (Self::I64(l), Self::I64(r)) => Self::I64(l + r),
            (Self::I64(l), Self::I32(r)) => Self::I64(l + r as i64),
            (Self::I64(l), Self::I16(r)) => Self::I64(l + r as i64),
            (Self::I64(l), Self::I8(r)) => Self::I64(l + r as i64),
            (Self::I32(l), Self::I32(r)) => Self::I32(l + r),
            (Self::I32(l), Self::I16(r)) => Self::I32(l + r as i32),
            (Self::I32(l), Self::I8(r)) => Self::I32(l + r as i32),
            (Self::I16(l), Self::I16(r)) => Self::I16(l + r),
            (Self::I16(l), Self::I8(r)) => Self::I16(l + r as i16),
            (Self::I8(l), Self::I8(r)) => Self::I8(l + r),
            (l, r) => r + l,
        }
    }
}

impl AddAssign for Int {
    fn add_assign(&mut self, other: Self) {
        let sum = *self + other;
        *self = sum;
    }
}

impl Sub for Int {
    type Output = Self;

    fn sub(self, other: Int) -> Self {
        match (self, other) {
            (Self::I64(l), Self::I64(r)) => Self::I64(l - r),
            (Self::I64(l), Self::I32(r)) => Self::I64(l - r as i64),
            (Self::I64(l), Self::I16(r)) => Self::I64(l - r as i64),
            (Self::I64(l), Self::I8(r)) => Self::I64(l - r as i64),
            (Self::I32(l), Self::I64(r)) => Self::I64(l as i64 - r),
            (Self::I32(l), Self::I32(r)) => Self::I32(l - r),
            (Self::I32(l), Self::I16(r)) => Self::I32(l - r as i32),
            (Self::I32(l), Self::I8(r)) => Self::I32(l - r as i32),
            (Self::I16(l), Self::I64(r)) => Self::I64(l as i64 - r),
            (Self::I16(l), Self::I32(r)) => Self::I32(l as i32 - r),
            (Self::I16(l), Self::I16(r)) => Self::I16(l - r),
            (Self::I16(l), Self::I8(r)) => Self::I16(l - r as i16),
            (Self::I8(l), Self::I64(r)) => Self::I64(l as i64 - r),
            (Self::I8(l), Self::I32(r)) => Self::I32(l as i32 - r),
            (Self::I8(l), Self::I16(r)) => Self::I16(l as i16 - r),
            (Self::I8(l), Self::I8(r)) => Self::I8(l - r),
        }
    }
}

impl SubAssign for Int {
    fn sub_assign(&mut self, other: Self) {
        let diff = *self - other;
        *self = diff;
    }
}

impl Sum for Int {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = IntType::Int.zero();
        for i in iter {
            sum = sum + i;
        }
        sum
    }
}

impl Mul for Int {
    type Output = Self;

    fn mul(self, other: Int) -> Self {
        match (self, other) {
            (Self::I64(l), Self::I64(r)) => Self::I64(l * r),
            (Self::I64(l), Self::I32(r)) => Self::I64(l * r as i64),
            (Self::I64(l), Self::I16(r)) => Self::I64(l * r as i64),
            (Self::I32(l), Self::I32(r)) => Self::I32(l * r),
            (Self::I32(l), Self::I16(r)) => Self::I32(l * r as i32),
            (Self::I16(l), Self::I16(r)) => Self::I16(l * r),
            (l, r) => r * l,
        }
    }
}

impl MulAssign for Int {
    fn mul_assign(&mut self, other: Self) {
        let product = *self * other;
        *self = product;
    }
}

impl Div for Int {
    type Output = Self;

    fn div(self, other: Int) -> Self {
        match (self, other) {
            (Self::I64(l), Self::I64(r)) => Self::I64(l / r),
            (Self::I64(l), Self::I32(r)) => Self::I64(l / r as i64),
            (Self::I64(l), Self::I16(r)) => Self::I64(l / r as i64),
            (Self::I64(l), Self::I8(r)) => Self::I64(l / r as i64),

            (Self::I32(l), Self::I64(r)) => Self::I64(l as i64 / r),
            (Self::I32(l), Self::I32(r)) => Self::I32(l / r),
            (Self::I32(l), Self::I16(r)) => Self::I32(l / r as i32),
            (Self::I32(l), Self::I8(r)) => Self::I32(l / r as i32),

            (Self::I16(l), Self::I64(r)) => Self::I64(l as i64 / r),
            (Self::I16(l), Self::I32(r)) => Self::I32(l as i32 / r),
            (Self::I16(l), Self::I16(r)) => Self::I16(l / r),
            (Self::I16(l), Self::I8(r)) => Self::I16(l / r as i16),

            (Self::I8(l), Self::I64(r)) => Self::I64(l as i64 / r),
            (Self::I8(l), Self::I32(r)) => Self::I32(l as i32 / r),
            (Self::I8(l), Self::I16(r)) => Self::I16(l as i16 / r),
            (Self::I8(l), Self::I8(r)) => Self::I8(l / r),
        }
    }
}

impl DivAssign for Int {
    fn div_assign(&mut self, other: Self) {
        let div = *self / other;
        *self = div;
    }
}

impl Product for Int {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = IntType::Int.zero();
        let mut product = IntType::Int.one();

        for i in iter {
            if i == zero {
                return zero;
            }

            product = product * i;
        }
        product
    }
}

impl PartialEq for Int {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I16(l), Self::I16(r)) => l.eq(r),
            (Self::I32(l), Self::I32(r)) => l.eq(r),
            (Self::I64(l), Self::I64(r)) => l.eq(r),
            (Self::I64(l), r) => l.eq(&i64::from(*r)),
            (l, r) => i64::from(*l).eq(&i64::from(*r)),
        }
    }
}

impl PartialOrd for Int {
    fn partial_cmp(&self, other: &Int) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Int {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Int::I16(l), Int::I16(r)) => l.cmp(r),
            (Int::I32(l), Int::I32(r)) => l.cmp(r),
            (Int::I64(l), Int::I64(r)) => l.cmp(r),
            (l, r) => i64::from(*l).cmp(&i64::from(*r)),
        }
    }
}

impl Default for Int {
    fn default() -> Int {
        Int::I16(i16::default())
    }
}

impl From<i8> for Int {
    fn from(i: i8) -> Int {
        Int::I8(i)
    }
}

impl From<i16> for Int {
    fn from(i: i16) -> Int {
        Int::I16(i)
    }
}

impl From<i32> for Int {
    fn from(i: i32) -> Int {
        Int::I32(i)
    }
}

impl From<i64> for Int {
    fn from(i: i64) -> Int {
        Int::I64(i)
    }
}

impl From<UInt> for Int {
    fn from(u: UInt) -> Int {
        match u {
            UInt::U64(u) => Int::I64(u as i64),
            UInt::U32(u) => Int::I32(u as i32),
            UInt::U16(u) => Int::I16(u as i16),
            UInt::U8(u) => Int::I16(u as i16),
        }
    }
}

impl From<Boolean> for Int {
    fn from(b: Boolean) -> Int {
        match b {
            Boolean(true) => Int::I16(1),
            Boolean(false) => Int::I16(0),
        }
    }
}

impl From<Int> for i64 {
    fn from(i: Int) -> i64 {
        match i {
            Int::I8(i) => i as i64,
            Int::I16(i) => i as i64,
            Int::I32(i) => i as i64,
            Int::I64(i) => i,
        }
    }
}

impl FromStr for Int {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        i64::from_str(s).map(Self::from).map_err(Error::new)
    }
}

impl<'de> Deserialize<'de> for Int {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        i64::deserialize(deserializer).map(Self::from)
    }
}

impl Serialize for Int {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Int::I8(i) => s.serialize_i8(*i),
            Int::I16(i) => s.serialize_i16(*i),
            Int::I32(i) => s.serialize_i32(*i),
            Int::I64(i) => s.serialize_i64(*i),
        }
    }
}

#[async_trait]
impl FromStream for Int {
    type Context = ();

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        i64::from_stream(cxt, decoder).map_ok(Self::from).await
    }
}

impl<'en> ToStream<'en> for Int {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match *self {
            Int::I8(i) => e.encode_i8(i),
            Int::I16(i) => e.encode_i16(i),
            Int::I32(i) => e.encode_i32(i),
            Int::I64(i) => e.encode_i64(i),
        }
    }
}

impl<'en> IntoStream<'en> for Int {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Int::I8(i) => e.encode_i8(i),
            Int::I16(i) => e.encode_i16(i),
            Int::I32(i) => e.encode_i32(i),
            Int::I64(i) => e.encode_i64(i),
        }
    }
}

impl fmt::Debug for Int {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.class(), self)
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Int::I8(i) => fmt::Display::fmt(i, f),
            Int::I16(i) => fmt::Display::fmt(i, f),
            Int::I32(i) => fmt::Display::fmt(i, f),
            Int::I64(i) => fmt::Display::fmt(i, f),
        }
    }
}

/// An unsigned integer.
#[derive(Clone, Copy, Hash)]
pub enum UInt {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

impl NumberInstance for UInt {
    type Abs = Self;
    type Exp = Float;
    type Class = UIntType;

    fn class(&self) -> UIntType {
        match self {
            Self::U8(_) => UIntType::U8,
            Self::U16(_) => UIntType::U16,
            Self::U32(_) => UIntType::U32,
            Self::U64(_) => UIntType::U64,
        }
    }

    fn into_type(self, dtype: UIntType) -> UInt {
        use UIntType::*;
        match dtype {
            U8 => match self {
                Self::U16(u) => Self::U8(u as u8),
                Self::U32(u) => Self::U8(u as u8),
                Self::U64(u) => Self::U8(u as u8),
                this => this,
            },
            U16 => match self {
                Self::U8(u) => Self::U16(u as u16),
                Self::U32(u) => Self::U16(u as u16),
                Self::U64(u) => Self::U16(u as u16),
                this => this,
            },
            U32 => match self {
                Self::U8(u) => Self::U32(u as u32),
                Self::U16(u) => Self::U32(u as u32),
                Self::U64(u) => Self::U32(u as u32),
                this => this,
            },
            U64 => match self {
                Self::U8(u) => Self::U64(u as u64),
                Self::U16(u) => Self::U64(u as u64),
                Self::U32(u) => Self::U64(u as u64),
                this => this,
            },
            UInt => self,
        }
    }

    fn abs(self) -> UInt {
        self
    }

    fn exp(self) -> Self::Exp {
        Float::from(self).exp()
    }

    fn pow(self, exp: Number) -> Self {
        if exp < Number::from(0) {
            return self.class().zero();
        }

        match self {
            Self::U8(this) => Self::U8(this.pow(exp.cast_into())),
            Self::U16(this) => Self::U16(this.pow(exp.cast_into())),
            Self::U32(this) => Self::U32(this.pow(exp.cast_into())),
            Self::U64(this) => Self::U64(this.pow(exp.cast_into())),
        }
    }
}

impl CastFrom<Complex> for UInt {
    fn cast_from(c: Complex) -> UInt {
        use Complex::*;
        match c {
            C32(c) => Self::U32(c.re as u32),
            C64(c) => Self::U64(c.re as u64),
        }
    }
}

impl CastFrom<Float> for UInt {
    fn cast_from(f: Float) -> UInt {
        use Float::*;
        match f {
            F32(f) => Self::U32(f as u32),
            F64(f) => Self::U64(f as u64),
        }
    }
}

impl CastFrom<Int> for UInt {
    fn cast_from(i: Int) -> UInt {
        use Int::*;
        match i {
            I8(i) => Self::U8(i as u8),
            I16(i) => Self::U16(i as u16),
            I32(i) => Self::U32(i as u32),
            I64(i) => Self::U64(i as u64),
        }
    }
}

impl CastFrom<UInt> for bool {
    fn cast_from(u: UInt) -> bool {
        use UInt::*;
        match u {
            U8(u) if u == 0u8 => false,
            U16(u) if u == 0u16 => false,
            U32(u) if u == 0u32 => false,
            U64(u) if u == 0u64 => false,
            _ => true,
        }
    }
}

impl CastFrom<UInt> for u8 {
    fn cast_from(u: UInt) -> u8 {
        use UInt::*;
        match u {
            U8(u) => u,
            U16(u) => u as u8,
            U32(u) => u as u8,
            U64(u) => u as u8,
        }
    }
}

impl CastFrom<UInt> for u16 {
    fn cast_from(u: UInt) -> u16 {
        use UInt::*;
        match u {
            U8(u) => u as u16,
            U16(u) => u,
            U32(u) => u as u16,
            U64(u) => u as u16,
        }
    }
}

impl CastFrom<UInt> for u32 {
    fn cast_from(u: UInt) -> u32 {
        use UInt::*;
        match u {
            U8(u) => u as u32,
            U16(u) => u as u32,
            U32(u) => u,
            U64(u) => u as u32,
        }
    }
}

impl Add for UInt {
    type Output = Self;

    fn add(self, other: UInt) -> Self {
        match (self, other) {
            (UInt::U64(l), UInt::U64(r)) => UInt::U64(l + r),
            (UInt::U64(l), UInt::U32(r)) => UInt::U64(l + r as u64),
            (UInt::U64(l), UInt::U16(r)) => UInt::U64(l + r as u64),
            (UInt::U64(l), UInt::U8(r)) => UInt::U64(l + r as u64),
            (UInt::U32(l), UInt::U32(r)) => UInt::U32(l + r),
            (UInt::U32(l), UInt::U16(r)) => UInt::U32(l + r as u32),
            (UInt::U32(l), UInt::U8(r)) => UInt::U32(l + r as u32),
            (UInt::U16(l), UInt::U16(r)) => UInt::U16(l + r),
            (UInt::U16(l), UInt::U8(r)) => UInt::U16(l + r as u16),
            (UInt::U8(l), UInt::U8(r)) => UInt::U8(l + r),
            (l, r) => r + l,
        }
    }
}

impl AddAssign for UInt {
    fn add_assign(&mut self, other: Self) {
        let sum = *self + other;
        *self = sum;
    }
}

impl Sub for UInt {
    type Output = Self;

    fn sub(self, other: UInt) -> Self {
        match (self, other) {
            (UInt::U64(l), UInt::U64(r)) => UInt::U64(l - r),
            (UInt::U64(l), UInt::U32(r)) => UInt::U64(l - r as u64),
            (UInt::U64(l), UInt::U16(r)) => UInt::U64(l - r as u64),
            (UInt::U64(l), UInt::U8(r)) => UInt::U64(l - r as u64),
            (UInt::U32(l), UInt::U32(r)) => UInt::U32(l - r),
            (UInt::U32(l), UInt::U16(r)) => UInt::U32(l - r as u32),
            (UInt::U32(l), UInt::U8(r)) => UInt::U32(l - r as u32),
            (UInt::U16(l), UInt::U16(r)) => UInt::U16(l - r),
            (UInt::U16(l), UInt::U8(r)) => UInt::U16(l - r as u16),
            (UInt::U8(l), UInt::U8(r)) => UInt::U8(l - r),
            (UInt::U8(l), UInt::U16(r)) => UInt::U16(l as u16 - r),
            (UInt::U8(l), UInt::U32(r)) => UInt::U32(l as u32 - r),
            (UInt::U8(l), UInt::U64(r)) => UInt::U64(l as u64 - r),
            (UInt::U16(l), r) => {
                let r: u64 = r.into();
                UInt::U16(l - r as u16)
            }
            (UInt::U32(l), r) => {
                let r: u64 = r.into();
                UInt::U32(l - r as u32)
            }
        }
    }
}

impl SubAssign for UInt {
    fn sub_assign(&mut self, other: Self) {
        let diff = *self - other;
        *self = diff;
    }
}

impl Sum for UInt {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = UIntType::UInt.zero();
        for i in iter {
            sum = sum + i;
        }
        sum
    }
}

impl Mul for UInt {
    type Output = Self;

    fn mul(self, other: UInt) -> Self {
        match (self, other) {
            (UInt::U64(l), UInt::U64(r)) => UInt::U64(l * r),
            (UInt::U64(l), UInt::U32(r)) => UInt::U64(l * r as u64),
            (UInt::U64(l), UInt::U16(r)) => UInt::U64(l * r as u64),
            (UInt::U64(l), UInt::U8(r)) => UInt::U64(l * r as u64),
            (UInt::U32(l), UInt::U32(r)) => UInt::U32(l * r),
            (UInt::U32(l), UInt::U16(r)) => UInt::U32(l * r as u32),
            (UInt::U32(l), UInt::U8(r)) => UInt::U32(l * r as u32),
            (UInt::U16(l), UInt::U16(r)) => UInt::U16(l * r),
            (UInt::U16(l), UInt::U8(r)) => UInt::U16(l * r as u16),
            (UInt::U8(l), UInt::U8(r)) => UInt::U8(l * r),
            (l, r) => r * l,
        }
    }
}

impl MulAssign for UInt {
    fn mul_assign(&mut self, other: Self) {
        let product = *self * other;
        *self = product;
    }
}

impl Div for UInt {
    type Output = Self;

    fn div(self, other: UInt) -> Self {
        match (self, other) {
            (UInt::U64(l), UInt::U64(r)) => UInt::U64(l / r),
            (UInt::U64(l), UInt::U32(r)) => UInt::U64(l / r as u64),
            (UInt::U64(l), UInt::U16(r)) => UInt::U64(l / r as u64),
            (UInt::U64(l), UInt::U8(r)) => UInt::U64(l / r as u64),

            (UInt::U32(l), UInt::U64(r)) => UInt::U64(l as u64 / r),
            (UInt::U32(l), UInt::U32(r)) => UInt::U32(l / r),
            (UInt::U32(l), UInt::U16(r)) => UInt::U32(l / r as u32),
            (UInt::U32(l), UInt::U8(r)) => UInt::U32(l / r as u32),

            (UInt::U16(l), UInt::U64(r)) => UInt::U64(l as u64 / r),
            (UInt::U16(l), UInt::U32(r)) => UInt::U32(l as u32 / r),
            (UInt::U16(l), UInt::U16(r)) => UInt::U16(l / r),
            (UInt::U16(l), UInt::U8(r)) => UInt::U16(l / r as u16),

            (UInt::U8(l), UInt::U64(r)) => UInt::U64(l as u64 / r),
            (UInt::U8(l), UInt::U32(r)) => UInt::U32(l as u32 / r),
            (UInt::U8(l), UInt::U16(r)) => UInt::U16(l as u16 / r),
            (UInt::U8(l), UInt::U8(r)) => UInt::U8(l / r),
        }
    }
}

impl DivAssign for UInt {
    fn div_assign(&mut self, other: Self) {
        let div = *self / other;
        *self = div;
    }
}

impl Product for UInt {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let zero = UIntType::UInt.zero();
        let mut product = UIntType::UInt.one();

        for i in iter {
            if i == zero {
                return zero;
            }

            product = product * i;
        }
        product
    }
}

impl Eq for UInt {}

impl PartialEq for UInt {
    fn eq(&self, other: &UInt) -> bool {
        match (self, other) {
            (Self::U8(l), Self::U8(r)) => l.eq(r),
            (Self::U16(l), Self::U16(r)) => l.eq(r),
            (Self::U32(l), Self::U32(r)) => l.eq(r),
            (Self::U64(l), Self::U64(r)) => l.eq(r),
            (l, r) => u64::from(*l).eq(&u64::from(*r)),
        }
    }
}

impl PartialOrd for UInt {
    fn partial_cmp(&self, other: &UInt) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for UInt {
    fn cmp(&self, other: &UInt) -> Ordering {
        match (self, other) {
            (Self::U8(l), Self::U8(r)) => l.cmp(r),
            (Self::U16(l), Self::U16(r)) => l.cmp(r),
            (Self::U32(l), Self::U32(r)) => l.cmp(r),
            (Self::U64(l), Self::U64(r)) => l.cmp(r),
            (l, r) => u64::from(*l).cmp(&u64::from(*r)),
        }
    }
}

impl Default for UInt {
    fn default() -> UInt {
        UInt::U8(u8::default())
    }
}

impl From<Boolean> for UInt {
    fn from(b: Boolean) -> UInt {
        match b {
            Boolean(true) => UInt::U8(1),
            Boolean(false) => UInt::U8(0),
        }
    }
}

impl From<u8> for UInt {
    fn from(u: u8) -> UInt {
        UInt::U8(u)
    }
}

impl From<u16> for UInt {
    fn from(u: u16) -> UInt {
        UInt::U16(u)
    }
}

impl From<u32> for UInt {
    fn from(u: u32) -> UInt {
        UInt::U32(u)
    }
}

impl From<u64> for UInt {
    fn from(u: u64) -> UInt {
        UInt::U64(u)
    }
}

impl From<UInt> for u64 {
    fn from(u: UInt) -> u64 {
        match u {
            UInt::U64(u) => u,
            UInt::U32(u) => u as u64,
            UInt::U16(u) => u as u64,
            UInt::U8(u) => u as u64,
        }
    }
}

impl From<UInt> for usize {
    fn from(u: UInt) -> usize {
        match u {
            UInt::U64(u) => u as usize,
            UInt::U32(u) => u as usize,
            UInt::U16(u) => u as usize,
            UInt::U8(u) => u as usize,
        }
    }
}

impl FromStr for UInt {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        u64::from_str(s).map(Self::from).map_err(Error::new)
    }
}

impl<'de> Deserialize<'de> for UInt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        u64::deserialize(deserializer).map(Self::from)
    }
}

impl Serialize for UInt {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            UInt::U8(u) => s.serialize_u8(*u),
            UInt::U16(u) => s.serialize_u16(*u),
            UInt::U32(u) => s.serialize_u32(*u),
            UInt::U64(u) => s.serialize_u64(*u),
        }
    }
}

#[async_trait]
impl FromStream for UInt {
    type Context = ();

    async fn from_stream<D: Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        u64::from_stream(cxt, decoder).map_ok(Self::from).await
    }
}

impl<'en> ToStream<'en> for UInt {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match *self {
            UInt::U8(u) => e.encode_u8(u),
            UInt::U16(u) => e.encode_u16(u),
            UInt::U32(u) => e.encode_u32(u),
            UInt::U64(u) => e.encode_u64(u),
        }
    }
}

impl<'en> IntoStream<'en> for UInt {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            UInt::U8(u) => e.encode_u8(u),
            UInt::U16(u) => e.encode_u16(u),
            UInt::U32(u) => e.encode_u32(u),
            UInt::U64(u) => e.encode_u64(u),
        }
    }
}

impl fmt::Debug for UInt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.class(), self)
    }
}

impl fmt::Display for UInt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UInt::U8(u) => fmt::Display::fmt(u, f),
            UInt::U16(u) => fmt::Display::fmt(u, f),
            UInt::U32(u) => fmt::Display::fmt(u, f),
            UInt::U64(u) => fmt::Display::fmt(u, f),
        }
    }
}
