let zero_int32 : int32 = (Int32.of_int 0)

let add_int32 (x: int32) (y: int32) : int32 = (Int32.add x y)

let sub_int32 (x: int32) (y: int32) : int32 = (Int32.sub x y)

let mul_int32 (x: int32) (y: int32) : int32 = (Int32.mul x y)

let div_int32 (x: int32) (y: int32) : int32 = (Int32.div x y)

let eq_int32 (x: int32) (y: int32) : bool = (x = y)

let lt_int32 (x: int32) (y: int32) : bool = x < y

let le_int32 (x: int32) (y: int32) : bool = x <= y

let gt_int32 (x: int32) (y: int32) : bool = x > y

let ge_int32 (x: int32) (y: int32) : bool = x >= y

type scalar_i32 = int32

