# Stage LRI
Integrales de choquet :
 - Data set :
    * Public
    * Entre 6 et 15 critères.
    * Entre 200 et 1600 donnés
 - En python :
    * Choquet de base
    * Ctypes pour importer des fonctions en C
 - Le réseau :
     * Vecteur X en entré
     * Réseau de neurone -> dit si une le choix est "bon" ou pas.
     * Extraction du réseau : les `ui` et les `wij` de par son architecture.
     * Regression :
        - testé avec des gaussiene
        - testé avec des sigmoides
     * Modele : trois sommes : \
        `wi.ui + wMij.max(ui,uj) + wmij.min(ui,uj)`
    * Probleme : \
        - Regression : on ne sait pas combien en mettre
 