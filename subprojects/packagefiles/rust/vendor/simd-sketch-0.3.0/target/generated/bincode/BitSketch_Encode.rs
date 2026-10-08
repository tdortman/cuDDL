impl :: bincode :: Encode for BitSketch
{
    fn encode < __E : :: bincode :: enc :: Encoder >
    (& self, encoder : & mut __E) ->core :: result :: Result < (), :: bincode
    :: error :: EncodeError >
    {
        match self
        {
            Self ::B32(field_0)
            =>{
                < u32 as :: bincode :: Encode >:: encode(& (0u32), encoder) ?
                ; :: bincode :: Encode :: encode(field_0, encoder) ?; core ::
                result :: Result :: Ok(())
            }, Self ::B16(field_0)
            =>{
                < u32 as :: bincode :: Encode >:: encode(& (1u32), encoder) ?
                ; :: bincode :: Encode :: encode(field_0, encoder) ?; core ::
                result :: Result :: Ok(())
            }, Self ::B8(field_0)
            =>{
                < u32 as :: bincode :: Encode >:: encode(& (2u32), encoder) ?
                ; :: bincode :: Encode :: encode(field_0, encoder) ?; core ::
                result :: Result :: Ok(())
            }, Self ::B1(field_0)
            =>{
                < u32 as :: bincode :: Encode >:: encode(& (3u32), encoder) ?
                ; :: bincode :: Encode :: encode(field_0, encoder) ?; core ::
                result :: Result :: Ok(())
            },
        }
    }
}