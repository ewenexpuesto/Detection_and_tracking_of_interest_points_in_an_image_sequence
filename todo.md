# Todo

* [ ] Utiliser le GPU ? NumPy ne fonctionne que sur le CPU mais pour OpenCV on peut le rendre utilisable par le GPU. Mais la combinaison des deux ne peut pas fonctionner avec le GPU
* [ ] UI to optimize the parameters, with real time visualization
* [ ] Documenter toutes les fonctions

Pour la difference of intensity of superpixels :

* [ ] Possible d'ajouter une heuristique pour d'abord réduire le temps de calcul : tous les N frames (donné en argument), il regarde s'il y a un cluster de superpixels qui passe le threashold : regarde si pour chaque superpixel qui passe le threshold il y a au moins V voisins, avec V la taille du côté du carré auquel il faut regarder autour qui passent le threshold. Si oui, il focus sur eux et ne calcule l'intensité moyenne que pour ces superpixels et les V+W autour (carré)
* [ ] Autre heuristique plus simple : si V (entier donné en argument) superpixels qui passent le threshold se touchent ensemble (en ligne, ou en colonne) et forment ainsi un cluster (tu regardes cela en parcourant chaque superpixel qui passe le threshold et pour ceux-là s'ils ont un voisin, tu regarde les voisins du voisin de manière récursive pour regarder s'ils passent le threshold). S'il y a un cluster dans la frame alors tu focus le calcul de l'intensité
* [ ] Pour la difference of intensity of superpixels, possible de relier les superpixels entre eux s'ils se touchent/sont proches et sinon supprimer les autres
* [ ] Possible aussi d'ajouter une limite du nombre de features (superpixels) détectées, en fonction de la taille de l'objet détecté
