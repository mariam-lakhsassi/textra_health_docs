.. role:: red
   :class: red

.. role:: green
   :class: green

.. role:: blue
   :class: blue

.. role:: orange
   :class: orange

.. raw:: html

   <style>
   .red {color: red;}
   .green {color: green;}
   .blue {color: blue;}
   .orange {color: orange;}
   </style>

Benchmarking des modèles OCR pour l'Extraction de Texte à partir d'Ordonnances Manuscrites
=========================================================================================

Introduction
------------
Ce document décrit le processus de benchmarking des modèles OCR (Optical Character Recognition) pour l'extraction de texte à partir d'ordonnances manuscrites. L'objectif principal était d'évaluer et de comparer les performances de différents modèles OCR sur un ensemble de données contenant des noms de médicaments manuscrits afin de sélectionner le modèle le plus adapté à notre cas d'utilisation.

**Objectifs**

- Évaluer les performances des modèles OCR disponibles sur des textes manuscrits.
- Comparer la précision, le rappel et le score F1 des différents modèles.
- Identifier le modèle OCR le plus performant pour notre ensemble de données spécifique.
- Documenter le processus et les résultats pour référence future.
.. raw:: html

   <a href=" https://colab.research.google.com/drive/1KCf9V_veW1ehjZRMrALcF4y-F3dCGGj5" target="_blank">
      <button style="background-color: #FF9E0B; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer;">
         Voir notebook dans Colab
      </button>
   </a>
Méthodologie
------------

Préparation des Données
~~~~~~~~~~~~~~~~~~~~~~~
- **Ensemble de données** : Nous avons utilisé un ensemble de données contenant des images de noms singuliers de médicaments manuscrits.
- **Format des images** : Les images étaient au format PNG, avec des variations de qualité, de résolution et de style d’écriture.
- **Annotations** : Chaque image était associée à une référence textuelle (le nom correct du médicament) pour évaluer la précision de l'OCR.

Processus d'Évaluation
~~~~~~~~~~~~~~~~~~~~~~
Pour chaque modèle OCR, nous avons :

- Chargé les images une par une.
- Appliqué l'OCR pour extraire le texte.
- Comparé le texte extrait avec la référence.
- Calculé les métriques de performance (précision, rappel, score F1) et le temps d'exécution.

Métriques Utilisées
~~~~~~~~~~~~~~~~~~~
- **Précision** : Proportion des mots extraits correctement parmi tous les mots extraits.
- **Rappel** : Proportion des mots extraits correctement parmi tous les mots attendus.
- **Score F1** : Moyenne harmonique de la précision et du rappel.
- **Temps d'exécution** : Temps nécessaire pour traiter une image.

Analyse Comparative des Performances des Modèles OCR
----------------------------------------------------
Nous avons évalué trois modèles OCR différents sur un ensemble de 780 images de noms de médicaments manuscrits. Voici une analyse détaillée des performances de chaque modèle :

PADDLEOCR
~~~~~~~~~

.. code-block:: none

   SUMMARY STATISTICS FOR PADDLEOCR:
   ---
   Number of files processed: 780
   Average Precision: 0.52
   Average Recall: 0.52
   Average F1 Score: 0.52
   Average Execution Time: 0.14 seconds per file

**Performances**

- Scores moyens de 0.52 pour la précision, le rappel et le F1.
- Performance modeste mais temps d'exécution très rapide (0.14s par image).

**Points forts**

- Solution la plus rapide de notre benchmark.
- Bonne option pour des applications nécessitant une réponse en temps réel.

**Points faibles**

- Exactitude globale relativement faible comparée aux autres modèles.
- Difficultés avec certaines formes d'écriture manuscrite.

TrOCR
~~~~~

.. code-block:: none

   SUMMARY STATISTICS:
   ---
   Number of files processed: 780
   Average Precision: 0.72
   Average Recall: 0.72
   Average F1 Score: 0.72
   Average Execution Time: 5.23 seconds per file

**Performances**

- Meilleurs scores avec 0.72 de précision, rappel et F1.
- Temps de traitement significativement plus long (5.23s par image).

**Points forts**

- Nette supériorité en termes de qualité de reconnaissance.
- Architecture basée sur les transformers bien adaptée au texte manuscrit.
- Meilleure résistance aux variations d'écriture.

**Points faibles**

- Lourdeur computationnelle importante.
- Temps de traitement plus long que PaddleOCR.

EasyOCR
~~~~~~~

.. code-block:: none

   SUMMARY STATISTICS:
   ---
   Number of files processed: 780
   Average Precision: 0.54
   Average Recall: 0.54
   Average F1 Score: 0.54
   Average Execution Time: 0.60 seconds per file

**Performances**

- Scores légèrement meilleurs que PaddleOCR (0.54).
- Temps d'exécution intermédiaire (0.60s).

**Points forts**

- Bon compromis vitesse/précision.
- Installation et utilisation simples.
- Performances stables.

**Points faibles**

- N'atteint pas la qualité de TrOCR.
- Temps de traitement supérieur à PaddleOCR.

Observations
------------

Erreurs Courantes
~~~~~~~~~~~~~~~~~
- Confusion entre des caractères similaires (par exemple, "e" et "a").
- Omission de caractères en fin de mot (par exemple, "Flugal" extrait comme "Fluga").
- Extraction de mots vides lorsque l'OCR échoue.

Analyse Comparative
~~~~~~~~~~~~~~~~~~~

1. **Précision de Reconnaissance** :
   - TrOCR domine clairement avec un F1 de 0.72.
   - L'écart entre EasyOCR et PaddleOCR est minime (0.54 vs 0.52).

2. **Vitesse d'Exécution** :
   - PaddleOCR est le plus rapide (0.14s), suivi de EasyOCR (0.60s).
   - TrOCR est significativement plus lent (5.23s) en raison de sa complexité.

Conclusion
----------
Ce benchmarking approfondi de ces trois modèles OCR révèle des compromis clairs entre précision et vitesse d'exécution sur notre corpus de 780 échantillons manuscrits. Les résultats obtenus nous orientent vers une implémentation hybride : EasyOCR comme solution principale pour sa polyvalence, complétée par TrOCR pour les cas complexes nécessitant une précision maximale, avec une optimisation via des techniques de prétraitement d'images et de post-traitement linguistique spécifique au domaine médical et pharmaceutique.
