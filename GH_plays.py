from github import Github
g = Github("Alexanu", "Theanswer1_")
for repo in g.get_user().get_repos():
    print(repo.parent.id)




g.get_user().get_repos()[2].id
g.get_user().get_repos()[2].parent.full_name
g.get_user().get_repos()[2].parent.id
g.get_user().get_repos()[2].parent.updated_at
g.get_user().get_repos()[2].description
g.get_user().get_repos()[2].created_at
g.get_user().get_repos()[2].size


import github3
for repo in GitHub.repositories_by("alexanu"):
    print('{0} created at {0.created_at}'.format(repo))

#all public repos
for repo in github3.all_repositories(number=10):
    print(repo)

import os
gh = github3.login(os.environ['GH_USERNAME'], os.environ['GH_PASSWORD'])

