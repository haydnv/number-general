use safecast::{CastFrom, CastInto};

pub mod class;
pub mod instance;

pub use class::*;
pub use instance::*;

type _Complex<T> = num_complex::Complex<T>;

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
