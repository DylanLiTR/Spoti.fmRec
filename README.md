# Spoti.fmRec
Recommender app using LastFM and Spotify.

### Things I've Tried
- Linear regression, polynomial regression, lasso regression, ridge regression

### Strengths
- Applied PCA and cross-validation
- Plotted 3D scatter with linear regression hyperplane

### Limitations
- Linear regression underfits complex music tastes
  - Almost no correlation between audio features and user playcount (0.05 coefficient of determination on 5 fold cross validation with my LastFM statistics)
