impl :: bincode :: Encode for BucketSketch
{
    fn encode < __E : :: bincode :: enc :: Encoder >
    (& self, encoder : & mut __E) ->core :: result :: Result < (), :: bincode
    :: error :: EncodeError >
    {
        :: bincode :: Encode :: encode(&self.rc, encoder) ?; :: bincode ::
        Encode :: encode(&self.k, encoder) ?; :: bincode :: Encode ::
        encode(&self.b, encoder) ?; :: bincode :: Encode ::
        encode(&self.buckets, encoder) ?; :: bincode :: Encode ::
        encode(&self.empty, encoder) ?; core :: result :: Result :: Ok(())
    }
}