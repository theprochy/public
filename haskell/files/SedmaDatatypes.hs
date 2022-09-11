module SedmaDatatypes where

data Suit = Heart | Diamond | Spade | Club deriving Show
data Rank = R7 | R8 | R9 | R10 | RJ | RQ | RK | RA deriving Show
data Card = Card Suit Rank deriving Show
type Cards = [Card]

data Team = AC | BD deriving (Show, Eq)
data Points = One | Two | Three deriving (Show, Eq)
type Winner = (Team, Points)

