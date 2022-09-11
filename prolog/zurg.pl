% LUP 2020/21 Prolog assignment
%
% Written on SWI-Prolog version 8.2.2
% To run the program it must be loaded by [zurg]. or passed as an
% argument when starting Prolog.
%
% This program solves the generalized version of the first problem: Escape
% from Zurg, the solution can be found just as shown in the example by running
% solve(T, L, S), where T is the given time limit and L is a list of arbitrary
% length of two member lists, the first element being the name of the toy
% and the second the time it takes is to cross. Prolog then finds the satisfying
% values of S, which are the solutions to the problem. The result format is
% just as in the example, it is a list which contains instructions in the format
% left_to_right or right_to_left and the one or two toys involved.
%
% Example from assignment: solve(60,[[buzz,5],[woody,10],[rex,20],[hamm,25]], S).
%
% BFS search is used by default, to switch to using DFS
% comment line 80 and uncomment line 81.

% maximum function
max(X, Y, Z) :- X >= Y, X = Z.
max(X, Y, Z) :- Y > X, Z = Y.
% X, Y are two different members of a list [H|T]
member2([H|T], X, Y) :- X = H, member(Y, T) ; member2(T, X, Y).

% arcs definition, we represent a node in the search space as a five member list,
% T is time remaining, L and R are toys on the left and right side of the
% bridge respecitvely, S represent the crossings of toys which have led to this
% state and the last member is a constant indicating where the flashlight is
% it is also possible to remove the third arc definition, since it is
% contraproductive for two toys to return, if there is some, there will be
% a faster solution, when only one toy returns. Currently, when given enough
% time to cross, the algorithm will return solutions containing passages of some
% toys back and forth on their own. However, these solutions do not seem invalid to me.
arc([T, L, R, S, l], [T2, L2, R2, S2, r]) :- T > 0, member2(L, X, Y),
  X = [X1, X2], Y = [Y1, Y2], max(X2, Y2, M), T2 is T - M,
  subtract(L, [X, Y], L2), append(R, [X, Y], R2), append([left_to_right(X1, Y1)], S, S2).
arc([T, L, R, S, l], [T2, L2, R2, S2, r]) :- T > 0, member(X, L),
  X = [X1, X2], T2 is T - X2,
  subtract(L, [X], L2), append(R, [X], R2), append([left_to_right(X1)], S, S2).
arc([T, L, R, S, r], [T2, L2, R2, S2, l]) :- T > 0, not(goal([T, L, R, S, r])), member2(R, X, Y),
  X = [X1, X2], Y = [Y1, Y2], max(X2, Y2, M), T2 is T - M,
  subtract(R, [X, Y], R2), append(L, [X, Y], L2), append([right_to_left(X1, Y1)], S, S2).
arc([T, L, R, S, r], [T2, L2, R2, S2, l]) :- T > 0, not(goal([T, L, R, S, r])), member(X, R),
  X = [X1, X2], T2 is T - X2,
  subtract(R, [X], R2), append(L, [X], L2), append([right_to_left(X1)], S, S2).
% goal definition, no toy left on the left side, flashlight is on the right
% and the crossing was made in time
goal([T, [], _, _, r]) :- T >= 0.

% Standard search algorithms declaration
children(Node, Children) :- findall(C, arc(Node, C), Children).

search_bf([Goal|_], PathRet, Goal) :-
  goal(Goal), Goal = [_, _, _, S, _], PathRet = S.
search_bf([Current|Rest], PathRet,  Goal) :-
  children(Current, Children),
  append(Rest, Children, NewAgenda),
  search_bf(NewAgenda, PathRet, Goal).

search_df([Goal|_], _, PathRet, Goal) :-
  goal(Goal), Goal = [_, _, _, S, _], PathRet = S.
search_df([Current|Rest], Visited, PathRet, Goal) :-
  children(Current, Children),
  add_df(Children, Rest, Visited, NewAgenda),
  search_df(NewAgenda, [Current|Visited], PathRet, Goal).
add_df([], Agenda, _, Agenda).
add_df([Child|Rest], OldAgenda, Visited, [Child|NewAgenda]) :-
  not(member(Child, OldAgenda)),
  not(member(Child, Visited)),
  add_df(Rest, OldAgenda, Visited, NewAgenda).
add_df([Child|Rest], OldAgenda, Visited, NewAgenda) :-
  member(Child, OldAgenda),
  add_df(Rest, OldAgenda, Visited, NewAgenda).
add_df([Child|Rest], OldAgenda, Visited, NewAgenda) :-
  member(Child, Visited),
  add_df(Rest, OldAgenda, Visited, NewAgenda).

solve(T, L, S) :- search_bf([[T, L, [], [], l]], S, _).
%solve(T, L, S) :- search_df([[T, L, [], [], l]], [], S, _).
