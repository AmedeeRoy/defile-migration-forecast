# Data

## Data description

### Species considered

- les cibles principales du suivi ont toujours été rapaces/ardéiformes/pigeons/corvidés ; le suivi des passereaux est hétérogène ; historiquement ils n'étaient presque pas noté ; durant la dernière décennie c'est un peu mieux mais le Défilé concentre peu les passereaux contrairement aux cols, et les observateurs ne sont pas toujours présent tôt le matin, donc l'exploitation des données est presque impossible.

### Data collected

- le dénominateur commun à chaque année depuis 1966 c'est un total/jour/espèce avec l'heure de début et heure de fin du suivi ; pour certaines années nous avons plus de détail (horaire) mais c'est hétérogène.
- les relevés météo n'ont jamais été numérisés, et rien n'était noté avant 2008.
- le nombre d'observateur actif par jour est noté de manière hétérogène depuis 2008 mais pas numérisé, donc la pression d'obs "réelle" n'est pas disponible
- les détails (age, sexe) ont été relevé de manière très hétérogène au fil du temps ; probablement exploitable pour le busard des roseaux depuis les années 2000, mais pour les autres espèces j'en doute

### Temporal coverage

- Le réel suivi quotidien a débuté en 1993, avant cela le suivi était plus ponctuel, très concentré sur les pigeons en octobre.
- Exception faite de 1983 et 1992, années pendant lesquelles la motivation de quelques observateurs a permis les premiers "vrai suivi".
- En 1993 le Dr Charvoz a commencé à suivre bénévolement tous les jours dès juillet, avec l'aide de J.P. Matérac, M. Maire et d'autres les week-end.
- Jusqu'en 2007 le suivi était assuré uniquement par les bénévoles.
- De 2008 à 2016 le suivi était assuré par un salarié de la LPO la semaine et par des bénévoles les week-end.
- Depuis 2017 le suivi est assurée par 2 salariés de la LPO du lundi au samedi et par des bénévoles les dimanches.

Nous avions saisit les données 1966-2007 du Dr Charvoz en décryptant au mieux ses fiches (écriture de médecin !) mais des infos se sont perdues.
Certains observateurs comme Lutz Lücker ont des souvenirs mémorables de migration des pigeons dont nous n'avons pas trace.

### Data collection

- de 2008 à 2016 on avait des fiches papiers standard pour noter par heure (heure locale), c'était donc saisie avec des totaux horaires. pour certaines journées il y a seulement un formulaire avec total jour
- de 2017 à 2020 on a de l'ultra-brute car saisie en direct avec Naturalist. donc pas d'heure de début et de fin de suivi dans les données (mais on a ça à coté), seulement l'heure de la saisie de la donnée, donc à quelques minutes près celle du passage des oiseaux "en majorité" car ce mode de saisie était appliqué la semaine par les spotteurs pour les journées assurées par les bénévoles il y a seulement un formulaire avec total jour.
- depuis 2021 on utilise l'appli Trektellen, faite pour le suivi de migration.

## Raw data available

### Pre-processing of the pre-2021 data

We are using the raw data `data/raw/all_data_défilé_tri_v2023` for all data until 2013 and `data/raw/data_brute_DE_2014_2021` for data between 2014 and 2021. A manual cleaning of this data was necessary as detailed below:

- Split data into (1) 1966-2013 providing daily count, (2) 2014-2016 providing hourly count based on data entered manually and (3) 2017-2021 data from Naturalist providing both list and manual entry.
- Fix startTime and endTime for 2014-2016:
  - 29.09.2014 et 11.10.2014 dans les données brut (data_brute_DE_2014_2021)
  - 09.10.2015, 01.12.2016 et 03.12.2016 dans pressure observation (all_data_défilé_tri_v2023)
- Fix and align startTime and endTime for 2017-2021
  - some sightings were providing without a list and without time. so probably seen during the day, but couldn't assign to a hour slot
  - 2-3 instances of interruption of list during the day: 29.10.2021, 17.11.2021 and 14.09.2017
  - Many case of sightings submitted before startTime or after endTime according to pressure observation. In most case I modified pressure observation, but in some case I deleted time (probably sumbmitted from home?)
  - pressure observation also had about 10 entries which seemed completly wrong, I removed those and use the last/first sightings.
- Delete observations after 19:00 for 2020-9-10 (pressure effort states 19:00 as end time, but there were 3 observations after.)
- Delete the observation of a Marsh Harrier on the 2020-11-03 at 20:05 because this if after endtime
- Delete observations oon the 2021-10-29: time of observations don't make sense.
- Modify time slightly to match end time for 22.Sep.17 19:04 -> 18:59, 14.Sep.19 20:00 -> 19:59, 22.Sep.19 20:00 -> 19:59, 2021-09-02 20:00, 2021-10-20 18:00, 2021-09-01 20:00

These modification and merging of the two dataset was performed manually into `data/count_2021.xlsx`

### Post-2021

We are using the year specific data file exported from Trektellen `data/raw/Trektellen_data_2422_{y}`.

## Processing

The processing of the count data, performed by `notebook/processing_count_data.ipynb`, combines the historical data until 2021 `data/count_2021.xlsx` and merge with the raw Trektellen files since 2021 `data/raw_count/Trektellen_data_2422_{y}.xlsx` to produce the file `all_count_processed.csv` which is used in the model.
