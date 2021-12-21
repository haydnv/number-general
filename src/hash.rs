use async_hash::Hash;
use sha2::digest::{Digest, Output};

use super::{Boolean, Complex, Float, Int, Number, UInt};

impl<D: Digest> Hash<D> for Number {
    fn hash(self) -> Output<D> {
        match self {
            Self::Bool(b) => Hash::<D>::hash(b),
            Self::Complex(c) => Hash::<D>::hash(c),
            Self::Float(f) => Hash::<D>::hash(f),
            Self::Int(i) => Hash::<D>::hash(i),
            Self::UInt(u) => Hash::<D>::hash(u),
        }
    }
}

impl<D: Digest> Hash<D> for Boolean {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(bool::from(self))
    }
}

impl<D: Digest> Hash<D> for Complex {
    fn hash(self) -> Output<D> {
        match self {
            Self::C32(c) => Hash::<D>::hash([c.re, c.im]),
            Self::C64(c) => Hash::<D>::hash([c.re, c.im]),
        }
    }
}

impl<D: Digest> Hash<D> for Float {
    fn hash(self) -> Output<D> {
        match self {
            Self::F32(f) => Hash::<D>::hash(f),
            Self::F64(f) => Hash::<D>::hash(f),
        }
    }
}

impl<D: Digest> Hash<D> for Int {
    fn hash(self) -> Output<D> {
        match self {
            Self::I8(i) => Hash::<D>::hash(i),
            Self::I16(i) => Hash::<D>::hash(i),
            Self::I32(i) => Hash::<D>::hash(i),
            Self::I64(i) => Hash::<D>::hash(i),
        }
    }
}

impl<D: Digest> Hash<D> for UInt {
    fn hash(self) -> Output<D> {
        match self {
            Self::U8(u) => Hash::<D>::hash(u),
            Self::U16(u) => Hash::<D>::hash(u),
            Self::U32(u) => Hash::<D>::hash(u),
            Self::U64(u) => Hash::<D>::hash(u),
        }
    }
}
