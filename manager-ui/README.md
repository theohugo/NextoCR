# Interface locale NextoCR

Ce sous-projet produit l'interface statique servie par le manager Python.
Le lancement normal de NextoCR ne dépend ni de Node.js ni d'un service cloud :
les fichiers prêts à servir sont versionnés dans `static/`.

Pour modifier l'interface :

```powershell
npm install
npm run typecheck
npm run build
```

Le build écrit `static/index.html` et ses assets relatifs. L'interface utilise
uniquement l'API locale de même origine sous `/api`.
