% Факты о ресурсах:
% resource(Имя, URL, Раздел, Ключи, Описание, Дата, Контакты)
resource('Wikipedia', 'https://wikipedia.org', 'Education', ['encyclopedia', 'knowledge', 'articles'], 'Free Encyclopedia', '2025-03-15', 'info@wikipedia.org').
resource('Khan Academy', 'https://khanacademy.org', 'Education', ['learning', 'math', 'science'], 'Free online courses', '2025-03-25', 'support@khanacademy.org').
resource('GitHub', 'https://github.com', 'IT', ['code', 'git', 'it'], 'Hosting for projects', '2026-01-10', 'info@github.com').
resource('Habr', 'https://habr.com', 'IT', ['it', 'programming', 'articles'], 'IT community', '2026-02-15', 'admin@habr.com').

% 1. Добавление ресурса
can_add_resource(N, U, S, K, D, Up, C) :-
    \+ resource(N, U, S, K, D, Up, C).

% 2. Поиск по разделу
search_by_section(Section, Name) :-
    resource(Name, _, Section, _, _, _, _).

% 3. Поиск по ключевому слову
search_by_keyword(Keyword, Name) :-
    resource(Name, _, _, Keywords, _, _, _),
    member(Keyword, Keywords).

% 4. Уточнение поиска (два слова)
refine_search(K1, K2, Name) :-
    resource(Name, _, _, Keywords, _, _, _),
    member(K1, Keywords),
    member(K2, Keywords).

% 5. Вывод полной информации (заменил writeln на write + nl)
show_full_info(Name) :-
    resource(Name, URL, Section, Keywords, Description, LastUpdate, Contact),
    write('========================'), nl,
    write('Name: '), write(Name), nl,
    write('URL: '), write(URL), nl,
    write('Section: '), write(Section), nl,
    write('Keywords: '), write(Keywords), nl,
    write('Description: '), write(Description), nl,
    write('Last Update: '), write(LastUpdate), nl,
    write('Contact: '), write(Contact), nl,
    write('========================'), nl.

% 6. Вывод краткой информации
show_short_info(Name) :-
    resource(Name, URL, _, _, _, _, _),
    write('========================'), nl,
    write('Name: '), write(Name), nl,
    write('URL: '), write(URL), nl,
    write('========================'), nl.

% 7. Фильтр по дате (GNU Prolog не любит sub_string, используем сопоставление)
% Вводи дату целиком '2026-01-10' или используй поиск по префиксу
filter_by_date(Date, Name) :-
    resource(Name, _, _, _, _, Date, _).