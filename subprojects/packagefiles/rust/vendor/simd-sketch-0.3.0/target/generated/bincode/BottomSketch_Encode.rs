impl :: bincode :: Encode for BottomSketch
{
    fn encode < __E : :: bincode :: enc :: Encoder >
    (& self, encoder : & mut __E) ->core :: result :: Result < (), :: bincode
    :: error :: EncodeError >
    {
        :: bincode :: Encode :: encode(&self.rc, encoder) ?; :: bincode ::
        Encode :: encode(&self.k, encoder) ?; :: bincode :: Encode ::
        encode(&self.bottom, encoder) ?; core :: result :: Result :: Ok(())
    }
}