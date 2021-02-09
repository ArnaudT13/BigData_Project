# Big Data Project
L’objectif du projet Big Data est d’étudier et de manipuler des extraits de CVs de différents individus dans le but de déterminer la bonne catégorie de poste correspondante.
Ce projet est séparé en cinq parties distinctes, dont quatre sont disponibles dans ce repository.

## Step 0 : Host - HDFS
Dans cette étape, les données mises à disposition doivent être transmises sur le système de fichier `HDFS` de la plateforme `Hadoop`. Aucun dossier ou ressource sont associés dans ce repository car les opérations ont été réalisées directement avec l'interface d'`Hadoop` `Ambari`.

## Step 1 : HDFS - Host
Le but de cette étape est de réaliser l'opération inverse de l'étape 0. Pour cela, deux scripts sont déployés : le premier `script_hdfs_hadoop.sh` permet de transmettre les données du système de fichier HDFS vers la machine `Hadoop` (VM linux) et le second `script_hadoop_windows.py` de la machine `Hadoop` vers le système hôte (Windows). Ils sont respectivement écrits en `Bash` et en `Python`.

## Step 2 : Host - VM AWS
Le transfert des données de la machine hôte vers la VM AWS est réalisé grâce au script `Python` `script_windows_aws.py`. Le caractère sensible des données est pris en compte, car le transport se fait à travers un tunnel sécurisé `SSH`. En effet, les protocoles sécurisés `SSH` et `SCP` sont utilisés pour que les données soient chiffrées de bout en bout.

## Step 3-4 : AWS ML
Les parties 3 et 4 sont placées dans le même dossier, car elles traitent toutes deux du domaine du Machine Learning. Le dossier est composé de trois sous-répertoires :  
- `Data` dans lequel sont placées les données à traiter.
- `Models` sotckant les modèles entraînés au format `joblib`.
- `NLP predicts` qui contient les scripts permettant de réaliser les prédictions à partir du fichier `predict.csv` situé dans `Predict Creation` et des modèles entraînés situés dans `Models`. Le fichier résultant de cette opération est placé dans ce même répertoire.
- `Notebook` où sont stockés les Notebooks détaillés et rédigés dans le but d'une prise de décision sur les méthodes de NLP à utiliser.
- `Predict creation` qui fournit un script de création du fichier `predict.csv` permettant d'assurer que les données à prédire n'ont pas été utilisées lors de la phase d'entraînement. Il utilise les données du sous-répertoire `Data` et stocke ce fichier dans ce même emplacement.
- `Train models` dans lequel sont présents les scripts d'entraînement des modèles. Leurs contenus sont tirés des résultats des Notebooks. De plus, le modèle est entraîné avec les données contenues dans le sous répertoire `Data` et la version entraînée est placée directement `Models`.

## Step 5 : VM AWS - MongoDB
Cette partie est réalisée par un seul script `script_aws_vm-mongodb.py`. Il permet dans un premier temps de rapatrier le fichier prédictions localement. Puis, il itère sur les lignes de ce fichier pour insérer une à une ligne dans la base de données `Mongo`. Il faudra préalablement créer la base de données sous le nom `bigdata_db` pour que la collection `predicts` puisse y être ajoutée.
