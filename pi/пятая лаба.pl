% Представление графа: edge(Откуда, Куда, Вес)

graph([
    edge(1, 2, 2), edge(1, 3, 3),
    edge(2, 3, 1), edge(2, 4, 1), edge(2, 5, 4),
    edge(3, 6, 5),
    edge(4, 5, 1),
    edge(5, 6, 1)
]).

% Вызов — prim(+StartNode, -MST)
prim(StartNode, MST) :-
    graph(Edges),
    % Инициализация: U = {StartNode}, P = {}
    solve_prim([StartNode], Edges, [], MST).

% Условие выхода (Конец): U = V (все вершины посещены).
solve_prim(Visited, Edges, AccMST, FinalMST) :-
    find_min_edge(Visited, Edges, BestEdge),
    !, % Если нашли ребро, продолжаем
    BestEdge = edge(U, V, W),
    % Добавляем новую вершину V в список посещенных
    % (т.к. ребро может быть направлено V->U, проверяем обе стороны)
    update_visited(U, V, Visited, NewVisited),
    solve_prim(NewVisited, Edges, [edge(U, V, W) | AccMST], FinalMST).

% Если минимальное ребро не найдено — алгоритм завершен.
solve_prim(_, _, MST, MST).

% Найти ребро (u,v) минимальной стоимости, где u ∈ Visited, а v ∉ Visited
find_min_edge(Visited, Edges, edge(U, V, W)) :-
    findall(edge(X, Y, Weight), (
        member(edge(X, Y, Weight), Edges),
        ((member(X, Visited), \+ member(Y, Visited), U = X, V = Y);
         (member(Y, Visited), \+ member(X, Visited), U = Y, V = X))
    ), Candidates),
    Candidates \= [],
    sort(3, @=<, Candidates, [edge(U, V, W) | _]).

% Обновление списка посещенных вершин
update_visited(U, V, Visited, [Next | Visited]) :-
    (member(U, Visited) -> Next = V ; Next = U).