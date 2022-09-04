# Mihajlo-RI
# Seminarski rad za kurs Racunarska Inteligencija 
# Tema rada je pronalazenja k klastera podataka minimalne sume

# Pokretanje:
U terminalu se pozicionirati unutar foldera od projekta i izvrsiti python fajl
python3 ./main.py 
kao argumente komandne linije moguce je proslediti putanju do fajla sa skupom podataka.
Ako nije prosledjena putanja do fajla algoritam ce sam generisati nasumicni dvodimenzionalni skup.  

| Argument | Opis | Vrednosti |
|:-----|:------:|:------:|
| -help         | izbacuje listu argumenata u terminalu  ||
| -metric={arg} | izbacuje listu argumenata u terminalu  | manhatn, euclid |
| -log          | ispisivanje rada algoritma u terminalu ||
| -fp={arg}     | preciznost zapisa realnog broja        | prirodan broj |
| -I            | interaktivni rezim rada                ||
| -k={arg}      | broj klustera za trazenje              | prirodan broj |
| -maxIter={arg}| maximalan broj iteracija               | prirodan broj |
