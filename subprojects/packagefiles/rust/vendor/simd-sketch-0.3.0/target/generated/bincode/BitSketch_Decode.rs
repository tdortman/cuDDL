impl < __Context > :: bincode :: Decode < __Context > for BitSketch
{
    fn decode < __D : :: bincode :: de :: Decoder < Context = __Context > >
    (decoder : & mut __D) ->core :: result :: Result < Self, :: bincode ::
    error :: DecodeError >
    {
        let variant_index = < u32 as :: bincode :: Decode ::< __D :: Context
        >>:: decode(decoder) ?; match variant_index
        {
            0u32 =>core :: result :: Result ::
            Ok(Self ::B32
            {
                0 : :: bincode :: Decode ::< __D :: Context >::
                decode(decoder) ?,
            }), 1u32 =>core :: result :: Result ::
            Ok(Self ::B16
            {
                0 : :: bincode :: Decode ::< __D :: Context >::
                decode(decoder) ?,
            }), 2u32 =>core :: result :: Result ::
            Ok(Self ::B8
            {
                0 : :: bincode :: Decode ::< __D :: Context >::
                decode(decoder) ?,
            }), 3u32 =>core :: result :: Result ::
            Ok(Self ::B1
            {
                0 : :: bincode :: Decode ::< __D :: Context >::
                decode(decoder) ?,
            }), variant =>core :: result :: Result ::
            Err(:: bincode :: error :: DecodeError :: UnexpectedVariant
            {
                found : variant, type_name : "BitSketch", allowed : &::
                bincode :: error :: AllowedEnumVariants :: Range
                { min: 0, max: 3 }
            })
        }
    }
} impl < '__de, __Context > :: bincode :: BorrowDecode < '__de, __Context >
for BitSketch
{
    fn borrow_decode < __D : :: bincode :: de :: BorrowDecoder < '__de,
    Context = __Context > > (decoder : & mut __D) ->core :: result :: Result <
    Self, :: bincode :: error :: DecodeError >
    {
        let variant_index = < u32 as :: bincode :: Decode ::< __D :: Context
        >>:: decode(decoder) ?; match variant_index
        {
            0u32 =>core :: result :: Result ::
            Ok(Self ::B32
            {
                0 : :: bincode :: BorrowDecode ::< __D :: Context >::
                borrow_decode(decoder) ?,
            }), 1u32 =>core :: result :: Result ::
            Ok(Self ::B16
            {
                0 : :: bincode :: BorrowDecode ::< __D :: Context >::
                borrow_decode(decoder) ?,
            }), 2u32 =>core :: result :: Result ::
            Ok(Self ::B8
            {
                0 : :: bincode :: BorrowDecode ::< __D :: Context >::
                borrow_decode(decoder) ?,
            }), 3u32 =>core :: result :: Result ::
            Ok(Self ::B1
            {
                0 : :: bincode :: BorrowDecode ::< __D :: Context >::
                borrow_decode(decoder) ?,
            }), variant =>core :: result :: Result ::
            Err(:: bincode :: error :: DecodeError :: UnexpectedVariant
            {
                found : variant, type_name : "BitSketch", allowed : &::
                bincode :: error :: AllowedEnumVariants :: Range
                { min: 0, max: 3 }
            })
        }
    }
}