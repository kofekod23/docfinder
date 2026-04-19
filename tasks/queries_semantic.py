"""Batterie de 130 requêtes sémantiques pour évaluer la pertinence du search.

Format : liste de (query_naturelle, path_substring_attendu).
- Les requêtes utilisent des synonymes et formulations "les documents qui parlent de..."
- On couvre à la fois les documents scannés (factures, feuilles, passeport, etc.)
  et les documents nativement numériques (audits, contrats, présentations).
- Le `path_substring_attendu` doit matcher (NFD-insensitive) `path` du résultat.
"""
from __future__ import annotations

QUERIES: list[tuple[str, str]] = [
    # ============================================================
    # SEO / Marketing digital (born-digital)
    # ============================================================
    ("les documents qui parlent d'audit de référencement naturel Tryba", "tryba"),
    ("analyse du positionnement SEO de Teddy Smith", "teddy smith"),
    ("rapport de crawl sur le site Teddy Smith", "teddy smith"),
    ("documents qui traitent de l'optimisation pour moteurs génératifs Louis Vuitton", "louis vuitton"),
    ("stratégie pour être visible dans les IA génératives pour Louis Vuitton", "louis vuitton"),
    ("documents sur la GEO generative engine optimization de Louis Vuitton", "louis vuitton"),
    ("présentations qui parlent du métier du référencement", "seo"),
    ("les documents qui parlent d'argumentation SEO Rakuten", "rakuten"),
    ("documents sur le trafic payant Rakuten Armando", "rakuten"),
    ("présentation sur mon parcours en marketing digital et data science", "parcours"),
    ("documents qui décrivent la recherche par vecteurs et similarité d'images", "mirror"),

    # ============================================================
    # Immobilier / Notaire (mix scanné + numérique)
    # ============================================================
    ("documents qui parlent de la promesse de vente Persoz", "persoz"),
    ("les documents qui parlent de vente immobilière signée avec Persoz", "persoz"),
    ("documents sur l'acquisition d'un appartement Persoz Giraud", "persoz"),
    ("attestation de remise de documents mutuelle prévoyance Rakuten", "attestation remise"),
    ("les papiers qui parlent d'une procuration pour acquérir un bien", "procuration"),
    ("documents qui expliquent la copropriété et les lots d'un immeuble", "persoz"),

    # ============================================================
    # Animal (Roy = chien Shar Pei, Peach = chat)
    # ============================================================
    ("documents qui parlent d'une feuille de soins pour un animal de compagnie", "roy"),
    ("les documents qui parlent de Roy le chien et de ses soins", "roy"),
    ("documents pour se faire rembourser des frais vétérinaires pour un chien", "roy"),
    ("assurance santé chien chat Carrefour pour mon animal", "carrefour assurance roy"),
    ("contrat d'assurance pour mon chien chez Carrefour", "carrefour assurance roy"),
    ("feuille de soin vierge à remplir pour un animal", "feuille de soin vierge"),
    ("documents qui parlent de la cession d'un chaton à une nouvelle famille", "peach"),
    ("attestation de cession d'un animal de compagnie", "peach"),
    ("certificat de vaccination pour mon chaton Peach", "peach"),
    ("facture vétérinaire pour le chat Peach", "peach"),

    # ============================================================
    # Santé / Sécurité sociale (scanné)
    # ============================================================
    ("les documents qui parlent d'une attestation de droits maladie ameli", "attestation-ameli"),
    ("mon attestation de sécurité sociale pour la CPAM", "attestation-ameli"),
    ("documents qui parlent d'un remboursement d'un tiers payant maladie", "paiementtiers"),
    ("justificatif de paiement par tiers à la sécurité sociale", "paiementtiers"),
    ("les factures qui parlent de lunettes et monture optique", "monture-verre"),
    ("facture d'opticien pour des verres correcteurs", "monture-verre"),
    ("garanties optique de l'assurance Axa santé", "optique axa"),
    ("la prévoyance Axa pour la famille", "prevoyance axa"),
    ("grille de garanties Axa 2020 pour les lunettes", "optique axa"),

    # ============================================================
    # Assurances / Mutuelles (scanné et numérique)
    # ============================================================
    ("les documents qui parlent de la mutuelle Rakuten", "mutuelle rakuten"),
    ("garanties santé de la complémentaire Rakuten", "mutuelle rakuten"),
    ("attestation de cumul d'assurances", "cumulatives"),
    ("je déclare sur l'honneur que je n'ai pas deux assurances", "cumulatives"),

    # ============================================================
    # Emploi / RH (mix)
    # ============================================================
    ("la lettre qui parle de ma démission chez Yakarouler", "demission yakarouler"),
    ("courrier de départ volontaire envoyé à Yakarouler", "demission yakarouler"),
    ("documents qui parlent d'un solde de tout compte chez Yakarouler", "solde-de-tout-compte"),
    ("récapitulatif des sommes dues à la fin du contrat Yakarouler", "solde-de-tout-compte"),
    ("bulletins de paie fiche de salaire Yakarouler", "bulletins-salaire"),
    ("mes fiches de paie chez Yakarouler", "bulletins-salaire"),
    ("les documents qui parlent de télétravail pour motif scolaire", "teletravail"),
    ("demande d'avenant télétravail à cause des enfants", "teletravail"),
    ("charte informatique et droit à la déconnexion", "charte informatique"),
    ("règlement sur le bon usage des outils numériques de l'entreprise", "charte informatique"),
    ("évaluation annuelle de mes performances 2022 avec Shugi", "year-end review"),
    ("mon bilan de fin d'année 2022 par le manager", "year-end review"),
    ("CV de directeur SEO ecommerce", "cv-directeur-seo"),
    ("mon curriculum vitae en tant que responsable SEO", "cv-directeur-seo"),
    ("lettre de motivation pour Hermès poste e-commerce", "hermes"),
    ("ma candidature pour un poste chez Hermès", "hermes"),
    ("évaluation wingfinder passeport de talents Red Bull", "passeport de talents"),
    ("les documents qui parlent de mes forces principales selon Wingfinder", "passeport de talents"),
    ("attestation signée par un ami en tant que témoin", "attestation de temoin"),
    ("attestations Julien et Sandrine pour faire foi", "attestations julien"),
    ("attestation manuscrite sur l'honneur", "attestation de temoin"),

    # ============================================================
    # Contestation / Juridique / Stationnement
    # ============================================================
    ("les documents qui parlent d'une contestation de stationnement à Paris", "rapo"),
    ("recours administratif préalable obligatoire pour un FPS à Paris", "rapo"),
    ("dossier complet de recours contre un forfait post-stationnement", "ccsp"),
    ("courrier au ministère public pour une majoration amende stationnement", "courrier omp"),
    ("contestation d'une amende de stationnement majorée", "courrier omp"),
    ("litige eBay acheteur livraison", "preuve conversation"),
    ("preuve conversation avec un acheteur eBay sur un carton livré", "preuve conversation"),
    ("les documents qui parlent de litiges sur eBay", "litiges"),

    # ============================================================
    # URSSAF / Fiscal (scanné)
    # ============================================================
    ("document de l'URSSAF Montreuil", "urssaf"),
    ("courrier Urssaf 93518 Montreuil", "urssaf"),
    ("document de synthèse INSEE URSSAF pour la création d'entreprise", "document_de_synthese"),
    ("avis de situation de l'INSEE d'avril 2026", "avis_de_situation"),
    ("conditions générales de vente pour la domiciliation d'entreprise", "cgv_voss"),
    ("les documents qui parlent de ma domiciliation d'entreprise", "domiciliation"),

    # ============================================================
    # Identité / Famille (scanné)
    # ============================================================
    ("les documents qui parlent du passeport de Victoire", "passeport victoire"),
    ("pièce d'identité de Victoire 2024", "passeport victoire"),
    ("convention de stage 3ème pour Victoire Giraud", "convention stage"),
    ("stage de découverte collège 3ème Victoire", "convention stage"),

    # ============================================================
    # Factures / Voyages (scanné)
    # ============================================================
    ("facture de l'hôtel Novotel en Égypte", "novotel"),
    ("hébergement Novotel Égypte facture", "novotel"),
    ("préautorisation Europcar et paiement de franchise location voiture", "pre-auth"),
    ("location voiture Europcar Paphos aéroport", "pre-auth"),

    # ============================================================
    # Banque / Assurance quotidienne (scanné)
    # ============================================================
    ("mandat SEPA pour l'assurance chat Carrefour", "sepa carrefour"),
    ("autorisation de prélèvement pour Carrefour assurance Roy", "sepa carrefour"),

    # ============================================================
    # Résiliations / Courrier administratif
    # ============================================================
    ("courrier de résiliation de l'abonnement Free box", "resiliation free"),
    ("lettre pour arrêter mon contrat internet Free", "resiliation free"),
    ("résiliation d'un emplacement de parking loué", "resiliation parking"),
    ("courrier pour libérer une place de parking", "resiliation parking"),

    # ============================================================
    # Prêt / Études
    # ============================================================
    ("les documents qui parlent d'un prêt pour les études de Victoire", "pret"),
    ("crédit étudiant pour financer une école", "pret"),

    # ============================================================
    # Divers personnel
    # ============================================================
    ("LAR place des finances pour valoir ce que de droit", "lar_place_des_finances"),
    ("clé de secours de mon compte et perte de mot de passe", "cle de secours"),
    ("un poème sur un coussin qui est doux comme de la mie de pain", "coussin"),
    ("documents ludiques ou poèmes personnels", "coussin"),

    # ============================================================
    # Scolaire / Questionnaire
    # ============================================================
    ("questionnaire donné au lycée par l'établissement", "questionnaire_lycee"),
    ("un formulaire rempli par un lycéen", "questionnaire_lycee"),
    ("séquence de cours sur la conception d'un médicament", "sequence"),
    ("synthèse d'un principe actif de médicament en cours de chimie", "sequence"),

    # ============================================================
    # Tryba (SEO + commerce) — queries par synonymes
    # ============================================================
    ("les documents qui parlent du site de fenêtres portes volets Tryba", "tryba"),
    ("recommandations SEO pour le fabricant de fenêtres Tryba", "tryba"),
    ("analyse de trafic et de visibilité du site Tryba", "tryba"),
    ("store locator géolocalisation des points de vente Tryba", "store locator"),
    ("documents qui parlent de la recherche d'un magasin Tryba près de chez moi", "store locator"),

    # ============================================================
    # LG / Rakuten / Contestations produits
    # ============================================================
    ("contestation d'un produit LG vendu sur Rakuten", "lg contestation"),
    ("litige avec LG concernant une commande Rakuten", "lg contestation"),

    # ============================================================
    # Contrats / Attestations Yakarouler
    # ============================================================
    ("attestation signée par Florent Thuilliez", "attestation-ft"),
    ("attestation d'un témoin habitant à Gennevilliers", "attestation-ft"),
    ("les documents qui parlent de Discount Auto Center Thiais", "yaka"),
    ("lettre reçue de Yakarouler en mars 2020", "yaka"),
    ("échange de mails concernant mon départ de Yakarouler", "yaka"),

    # ============================================================
    # Argumentations Rakuten (spécifique : pptx corrompu et pdf)
    # ============================================================
    ("argumentations diverses sur Rakuten", "argumentations"),
    ("document avec des arguments défendus lors d'une réunion Rakuten", "argumentations"),

    # ============================================================
    # Requêtes à intention large / scannés difficiles
    # ============================================================
    ("documents officiels scannés sur des démarches administratives", "urssaf"),
    ("factures et reçus de voyage à l'étranger", "novotel"),
    ("documents avec des tableaux Excel de visibilité SEO", "visibilite"),
    ("fichier Excel avec les pages indexées par Google", "pages indexees"),
    ("tableau des prix de pièces automobiles Oscaro Knecht", "releve"),
    ("un relevé Excel de comparaison de prix de pièces auto", "releve"),
    ("fichier Excel avec les ventes de pièces détachées auto", "releve"),

    # ============================================================
    # Variations plus exigeantes (test robustesse sémantique)
    # ============================================================
    ("je cherche la présentation GEO Louis Vuitton en anglais", "geo-louis-vuitton (en)"),
    ("la version française du plan d'action GEO pour Louis Vuitton", "geo-louis-vuitton fr"),
    ("le document qui présente l'architecture hybride CSR SSR pour Louis Vuitton", "louis vuitton"),
    ("le compte rendu d'échange avec Thibaud Etienne", "thibaud etienne"),
    ("entretien avec le responsable d'acquisition du portefeuille", "thibaud etienne"),

    # ============================================================
    # Paraphrases du top 20 (pour comparer avec test_targeted.py)
    # ============================================================
    ("retrouve l'audit SEO de Tryba", "audit seo"),
    ("la promesse de vente avec Persoz et Giraud", "promesse de vente"),
    ("je veux mon CV de directeur SEO", "cv-directeur-seo"),
    ("contrat télétravail avec motif scolaire", "teletravail"),
    ("feuille de soin pour mon animal nommé Roy", "feuille de soin"),
    ("mes bulletins de paie chez Yakarouler", "bulletins-salaire"),
    ("procuration pour acquérir Persoz", "procuration"),
    ("le document de la mutuelle Rakuten de 2020", "mutuelle rakuten"),
    ("la grille de prévoyance Axa", "prevoyance axa"),
    ("le store locator Tryba", "store locator"),
    ("le solde de tout compte", "solde-de-tout-compte"),
    ("la présentation Louis Vuitton sur le GEO", "geo-louis-vuitton"),
    ("le document d'argumentations Rakuten", "argumentations"),
    ("la charte informatique et la déconnexion", "charte informatique"),
    ("la facture du Novotel en Égypte", "novotel"),
    ("le questionnaire du lycée", "questionnaire_lycee"),
    ("la contestation LG sur Rakuten", "lg contestation"),
    ("un prêt pour les études", "pret"),
    ("le SEPA de Carrefour assurance chat", "sepa carrefour"),
    ("les litiges ebay", "litiges"),
]
