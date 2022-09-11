import SedmaDatatypes

type State = ((Int, Int), (Bool, Bool), Int)
type TrickState = ([Int], Int, Int, Rank)

instance Eq Suit where
    Heart == Heart = True
    Spade == Spade = True
    Diamond == Diamond = True
    Club == Club = True
    _ == _ = False

instance Eq Rank where
    R7 == R7 = True
    R8 == R8 = True
    R9 == R9 = True
    R10 == R10 = True
    RJ == RJ = True
    RQ == RQ = True
    RK == RK = True
    RA == RA = True
    _ == _ = False

instance Eq Card where
    (Card s1 r1) == (Card s2 r2) = (s1 == s2) && (r1 == r2)

get_rank :: Card -> Rank
get_rank (Card s r) = r

points_in_hand :: Int -> Card -> Int
points_in_hand points card =
    if (get_rank card) == R10 || (get_rank card) == RA
        then points + 10
        else points

get_cur_winner :: Int -> Int -> Rank -> Rank -> Int
get_cur_winner cur new cur_c new_c =
    if cur_c == new_c || new_c == R7
        then new
        else cur

first3 (x,_,_) = x
second3 (_,x,_) = x
third3 (_,_,x) = x
first4 (x,_,_,_) = x
second4 (_,x,_,_) = x
third4 (_,_,x,_) = x
fourth4 (_,_,_,x) = x

valid_match :: Cards -> Bool
valid_match cards | null cards = False
                  | length cards /= 32 = False
                  | otherwise = valid_match_h cards

valid_match_h :: Cards -> Bool
valid_match_h cards =
    if null cards
        then True
        else
            if elem (head cards) (tail cards)
                then False
                else valid_match_h (tail cards)

replay :: Cards -> Maybe Winner
replay cards | not (valid_match cards) = Nothing
             | otherwise = replay_h (cards, ((0,0), (False,False), 0))

replay_h :: (Cards, State) -> Maybe Winner
replay_h input =
    let cards = fst input
        scores = first3 (snd input)
        won_trick = second3 (snd input)
        to_play = third3 (snd input)
    in
        if null cards
            then
            if to_play == 0 || to_play == 2
                then get_winner ((fst scores) + 10, snd scores) won_trick
                else get_winner (fst scores, (snd scores) + 10) won_trick
            else replay_h (drop 4 cards,
                           play_trick(take 4 cards,
                                      snd input,
                                      ([mod (to_play + x) 4 | x <- [0..3]],0,to_play,R7))) 

play_trick :: (Cards, State, TrickState) -> State
play_trick input =
    let cards = first3 input
        scores = first3 (second3 input)
        won_trick = second3 (second3 input)
        players = first4 (third3 input)
        points = second4 (third3 input)
        cur_winner = third4 (third3 input)
        cur_card = fourth4 (third3 input)
    in
        if null cards
            then if cur_winner == 0 || cur_winner == 2
                    then (((fst scores) + points, snd scores),
                          (True, snd won_trick),
                          cur_winner)
                    else ((fst scores, (snd scores) + points),
                          (fst won_trick, True),
                          cur_winner)
            else if length players == 4
                    then play_trick(drop 1 cards,
                                    second3 input,
                                    (drop 1 players,
                                     points_in_hand points (head cards),
                                     head players,
                                     get_rank (head cards)))
            else
                play_trick(drop 1 cards,
                           second3 input,
                           (drop 1 players,
                            points_in_hand points (head cards),
                            get_cur_winner cur_winner (head players) cur_card (get_rank (head cards)),
                            cur_card))   
                

get_winner :: (Int, Int) -> (Bool, Bool) -> Maybe Winner
get_winner scores won_trick =
    if (fst scores) > (snd scores)
        then if (snd scores) == 0
                then if not (snd won_trick)
                        then Just (AC, Three)
                        else Just (AC, Two)
                else Just (AC, One)
        else if (fst scores) == 0
                then if not (fst won_trick)
                        then Just (BD, Three)
                        else Just (BD, Two)
                else Just (BD, One)

