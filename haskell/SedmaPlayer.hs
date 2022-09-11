import SedmaBase
import SedmaReplay
import SedmaGamble

data AIPlayerState = AIPlayerState Player Cards Cards

getJust :: Maybe a -> a
getJust (Just a) = a

removeElem _ []                 = []
removeElem e (x:xs) | e == x    = removeElem e xs
                    | otherwise = x : removeElem e xs

sameRank :: Card -> Card -> Bool
sameRank (Card s1 r1) (Card s2 r2) = r1 == r2

newHand :: Hand -> Card -> Maybe Card -> Hand
newHand hand playedCard Nothing  = (removeElem playedCard hand) ++ []
newHand hand playedCard (Just c) = (removeElem playedCard hand) ++ [c]

getSame :: Hand -> Card -> Maybe Card
getSame [] _ = Nothing
getSame (x:xs) card | sameRank x card = Just x
                    | otherwise = getSame xs card

getSeven :: Hand -> Maybe Card
getSeven [] = Nothing
getSeven ((Card s r):xs) | r == R7 = Just (Card s r)
                         | True = Nothing

getWinner :: Int -> Int -> Card -> Hand -> Int
getWinner winner _ _ [] = winner
getWinner winner i card (x:xs)
    | sameRank card x = getWinner i (i+1) card xs
    | True = getWinner winner (i+1) card xs

try10orA :: Hand -> Maybe Card
try10orA [] = Nothing
try10orA ((Card s r):xs) | (r == R10) || (r == RA) = Just (Card s r)
                         | True = try10orA xs

tryNot7 :: Hand -> Maybe Card
tryNot7 [] = Nothing
tryNot7 ((Card s r):xs) | r /= R7 = Just (Card s r)
                        | True = tryNot7 xs

tryNotImportant :: Hand -> Maybe Card
tryNotImportant [] = Nothing
tryNotImportant ((Card s r):xs) | (r /= R7) && (r /= R10) && (r /= RA) = Just (Card s r)
                                | True = tryNotImportant xs

playedCount :: Int -> Card -> Cards -> Int
playedCount i _ [] = i
playedCount i card (x:xs)
    | sameRank card x = playedCount (i+1) card xs
    | True = playedCount i card xs 

tryLast :: Hand -> Cards -> Maybe Card
tryLast [] played = Nothing
tryLast (card:rest) played
    | (playedCount 0 card played) == 3 = Just card
    | True = tryLast rest played

tryMatch :: Card -> Hand -> Maybe Card
tryMatch card hand
    | not (getSame hand card == Nothing)
        = getSame hand card
    | not (getSeven hand == Nothing)
        = getSeven hand
    | otherwise = Nothing

walue :: Card -> Int
walue (Card _ R10) = 10
walue (Card _ RA) = 10
walue (Card _ _) = 0

trickValue :: Trick -> Int
trickValue [] = 0
trickValue (x:xs) = (walue x) + trickValue xs 

instance PlayerState AIPlayerState where
  initState name hand = AIPlayerState name hand []
  updateState trick firstPlayer playedCard newCard (AIPlayerState name hand played)
    = AIPlayerState name (newHand hand playedCard newCard) (played ++ [playedCard])

player :: AIPlayer AIPlayerState
player trick (AIPlayerState player hand played) =
    case length trick of
        0 -> if tryLast hand played == Nothing
                 then if tryNotImportant hand == Nothing
                         then head hand
                         else getJust (tryNotImportant hand)
                 else getJust (tryLast hand played)

        1 -> if (tryMatch (head trick) hand) == Nothing
                 then if tryNotImportant hand == Nothing
                          then head hand
                          else getJust (tryNotImportant hand)
                 else getJust (tryMatch (head trick) hand)

        2 -> if (getWinner 0 1 (head trick) (tail trick)) == 1
                then if try10orA hand == Nothing
                         then if (tryNot7 hand) == Nothing
                                  then (head hand)
                                  else getJust (tryNot7 hand)
                         else getJust (try10orA hand)
                
                else if trickValue trick > 0
                        then if (tryMatch (head trick) hand) == Nothing
                                then head hand
                                else getJust (tryMatch (head trick) hand)
                        else if tryNotImportant hand == Nothing
                                then head hand
                                else getJust (tryNotImportant hand)

        3 -> if (getWinner 0 1 (head trick) (tail trick)) == 1
                then if try10orA hand == Nothing
                         then if (tryNot7 hand) == Nothing
                                  then (head hand)
                                  else getJust (tryNot7 hand)
                         else getJust (try10orA hand)
                
                else if trickValue trick > 0
                        then if (tryMatch (head trick) hand) == Nothing
                                then head hand
                                else getJust (tryMatch (head trick) hand)
                        else if tryNotImportant hand == Nothing
                                then head hand
                                else getJust (tryNotImportant hand)
