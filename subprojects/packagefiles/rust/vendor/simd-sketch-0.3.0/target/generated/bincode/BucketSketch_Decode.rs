impl < __Context > :: bincode :: Decode < __Context > for BucketSketch
{
    fn decode < __D : :: bincode :: de :: Decoder < Context = __Context > >
    (decoder : & mut __D) ->core :: result :: Result < Self, :: bincode ::
    error :: DecodeError >
    {
        core :: result :: Result ::
        Ok(Self
        {
            rc : :: bincode :: Decode :: decode(decoder) ?, k : :: bincode ::
            Decode :: decode(decoder) ?, b : :: bincode :: Decode ::
            decode(decoder) ?, buckets : :: bincode :: Decode ::
            decode(decoder) ?, empty : :: bincode :: Decode :: decode(decoder)
            ?,
        })
    }
} impl < '__de, __Context > :: bincode :: BorrowDecode < '__de, __Context >
for BucketSketch
{
    fn borrow_decode < __D : :: bincode :: de :: BorrowDecoder < '__de,
    Context = __Context > > (decoder : & mut __D) ->core :: result :: Result <
    Self, :: bincode :: error :: DecodeError >
    {
        core :: result :: Result ::
        Ok(Self
        {
            rc : :: bincode :: BorrowDecode ::< '_, __Context >::
            borrow_decode(decoder) ?, k : :: bincode :: BorrowDecode ::< '_,
            __Context >:: borrow_decode(decoder) ?, b : :: bincode ::
            BorrowDecode ::< '_, __Context >:: borrow_decode(decoder) ?,
            buckets : :: bincode :: BorrowDecode ::< '_, __Context >::
            borrow_decode(decoder) ?, empty : :: bincode :: BorrowDecode ::<
            '_, __Context >:: borrow_decode(decoder) ?,
        })
    }
}