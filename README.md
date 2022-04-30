## Вычислительный эксперимент

Решение краевой задачи Пуассона в прямоугольной области с граничными условиями Дирихле

#### Алгоритмы для сравнения

* Прямой (нахождение матрицы K и решение системы уравнений в явном виде)
* Параллельный в n процессов (разбиение матрицы K на куски с последующей параллельной обработкой)
* Один поток с разбиением матрицы K

#### Конфигурация машины

```
Model Name:	            MacBook Pro
Model Identifier:	    MacBookPro17,1
Chip:	                    Apple M1
Total Number of Cores:	    8 (4 performance and 4 efficiency)
Memory:	                    8 GB
```

#### Сравнение с теоретическими результатами

Оценка качества производилось путем проверки идентичности результатов параллельного алгоритма значениям полученным с помощью наивного алгоритма. 

#### Таблица с результатами сравнения 

Значения времени приведены в микросекндах. Все значения из таблицы получены как медианные на 5 запусках алгоритма.

| Размер грида | Количество частей (влияет только на параллельные) | Прямой | Параллельный (2 процесса) | Параллельный (4 процесса) | Параллельный (8 процессов) | Один поток |
|--------------|---------------------------------------------------|--------|---------------------------|---------------------------|----------------------------|------------|
| 3x7 | 2 | 44     | 297111 | 453073 | 586933 | 76 |
| 3x7 | 4 | 44     | 205344 | 449109 | 543818 | 352 |
| 11x23 | 2 | 1600   | 347249 | 431806 | 764346 | 1174 |
| 11x23 | 4 | 1600   | 408343 | 492470 | 897737 | 9376 |
| 11x23 | 6 | 1600   | 346312 | 500997 | 603168 | 8456 |
| 11x23 | 8 | 1600   | 297938 | 514952 | 677459 | 1210 |
| 23x47 | 2 | 48425  | 923878 | 161931 | 488551 | 11126 |
| 23x47 | 4 | 48425  | 681318 | 228316 | 640313 | 2950 |
| 23x47 | 6 | 48425  | 134819 | 582038 | 775323 | 3046 |
| 23x47 | 8 | 48425  | 248758 | 646581 | 563799 | 3536 |
| 47x95 | 2 | 620423 | 231437 | 621571 | 384298 | 284377 |
| 47x95 | 4 | 620423 | 258818 | 333242 | 668608 | 105292 |
| 47x95 | 8 | 620423 | 110691 | 620113 | 610211 | 63894 |
| 47x95 | 16 | 620423 | 729801 | 360226 | 483153 | 174629 |
| 47x95 | 32 | 620423 | 786640 | 327761 | 260471 | 354120 |
| 30x255 | 2 | 374781 | 812416 | 682978 | 912139 | 153318 |
| 30x255 | 16 | 374781 | 429540 | 303314 | 519595 | 138481 |
| 30x255 | 64 | 374781 | 442888 | 211467 | 892050 | 940740 |
| 20x511 | 64 | 791036 | 794226 | 433140 | 657799 | 637718 |
| 20x511 | 128 | 791036 | 256213 | 281427 | 468449 | 552486 |
| 20x1023 | 128 | 618300 | 324065 | 360311 | 812562 | 379097 |


### Выводы

Основное узкое место даннной реализации - передача больших объектов между процессами. Возможные исправления:

* Переписать код на другой язык с параллелизмом через потоки
* Хранить матрицы в разреженном виде
* Воспользоваться другим интерпретатором Python без GIL.

Помимо этого еще одно значительное узкое место этой реализации в том, что параллелизм возможен только по оси X. В ином случае представленный алгоритм из книги "A Tutorial on Elliptic PDE Solvers and their Parallelization" нуждался бы в модификации.


                                                 