# Big Data Project
L’objectif du projet Big Data est d’étudier et de manipuler des extraits de CVs de différents individus dans le but de déterminer la bonne catégorie de poste correspondante.
Ce projet est séparé en cinq parties distinctes, dont quatre sont disponibles dans ce repository.


## Step 0 : Host - HDFS
Dans cette étape, les données mises à disposition doivent être transmises sur le système de fichier `HDFS` de la plateforme `Hadoop`. Aucun dossier ou ressource sont associés dans ce repository car les opérations ont été réalisées directement avec l'interface d'`Hadoop` `Ambari`.  
L’environnement Hadoop constitue le point de départ de notre projet.


## Step 1 : HDFS - Host
Le but de cette étape est de réaliser l'opération inverse de l'étape 0, qui est le transfert depuis le système `HDFS` vers le host Windows en passant par la VM `Haddop`. Pour cela, deux scripts sont déployés : le premier `script_hdfs_hadoop.sh` permet de transmettre les données du système de fichier HDFS vers la machine `Hadoop` (VM linux) et le second `script_hadoop_windows.py` de la machine `Hadoop` vers le système hôte (Windows). Ils sont respectivement écrits en `Bash` et en `Python`.


## Step 2 : Host - VM AWS
Le transfert des données de la machine hôte vers la VM AWS est réalisé grâce au script `Python` `script_windows_aws.py`. Le caractère sensible des données est pris en compte, car le transport se fait à travers un tunnel sécurisé `SSH`. En effet, les protocoles sécurisés `SSH` et `SCP` sont utilisés pour que les données soient chiffrées de bout en bout.  
La VM AWS utilisée est une instance `EC2` avec 32Go de RAM et 8 coeurs (ces propriétés permettent seulement de réduire le temps de calcul).


## Step 3-4 : AWS ML
### Organisation des répertoires
Les parties 3 et 4 sont placées dans le même dossier, car elles traitent toutes deux du domaine du Machine Learning. Le dossier est composé de trois sous-répertoires :  
- `Data` dans lequel sont placées les données à traiter.
- `Models` stockant les modèles entraînés au format `joblib`.
- `NLP predicts` qui contient les scripts permettant de réaliser les prédictions à partir du fichier `predict.csv` situé dans `Predict Creation` et des modèles entraînés situés dans `Models`. Le fichier résultant de cette opération est placé dans ce même répertoire.
- `Notebook` où sont stockés les Notebooks détaillés et rédigés dans le but d'une prise de décision sur les méthodes de NLP à utiliser.
- `Predict creation` qui fournit un script de création du fichier `predict.csv` permettant d'assurer que les données à prédire n'ont pas été utilisées lors de la phase d'entraînement. Il utilise les données du sous-répertoire `Data` et stocke ce fichier dans ce même emplacement.
- `Train models` dans lequel sont présents les scripts d'entraînement des modèles. Leurs contenus sont tirés des résultats des Notebooks. De plus, le modèle est entraîné avec les données contenues dans le sous-répertoire `Data` et la version entraînée est placée directement `Models`.

### Les Notebooks
Pour le choix de la technologie de Machine Learning, nous avons utilisé `Jupyter`. En effet, nous l'avons préalablement installé sur la VM AWS car celui-ci n'est pas présent initialement.  
Toutes nos recherches sont détaillées et expliquées dans les trois Notebooks suivants (dans l'ordre de création):
- `Notebook_ML_Count-TFIDF_Chi2.ipynb` qui propose un préprocessing des données, une vectorisation avec `Count` ou `TF-IDF`, une sélection des features avec la méthode du `Chi2` et une comparaison de la précision des différents classifiers sur cette base (mise en place de pipelines).
- `Notebook_ML_Count-TFIDF_SVD.ipynb` qui propose la même chose que le Notebook précédent, mais avec une sélection des features réalisée avec `SVD` (réduction de dimensions).
- `Notebook_ML_Word2Vec_Doc2Vec.ipynb` qui propose un préprocessing, l'application de `Word2Vec` ou `Doc2Vec` ainsi qu'une comparaison de la précision des différents classifiers.

### Obtention des prédictions
Pour obtenir les prédictions, nous avons réalisé les étapes suivantes :
- Création du fichier `predict.csv` grâce au script `create_predict_csv.py`
- Entrainement des modèles suivant la technologie (environ 1h) via les scripts de `Train models` et exportation du modèle dans `Models`
- Réalisation des prédictions à partir du modèle obtenu précédemment (environ 5 min)
Pour vérifier les résultats et obtenir des statistiques nous avons importé les données dans un fichier Excel nommé `Prediction_score.xlsx`.  


## Step 5 : VM AWS - MongoDB
Cette dernière partie est réalisée par un seul script `script_aws_vm-mongodb.py`. Il permet dans un premier temps de rapatrier le fichier de prédictions localement. Puis, il itère sur les lignes de ce fichier pour insérer une à une ligne dans la base de données `Mongo`. Il faudra préalablement créer la base de données sous le nom `bigdata_db` pour que la collection `predicts` puisse y être ajoutée.
