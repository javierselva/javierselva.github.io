---
layout: post
section-type: post
has-comments: false
title: "Setting up GitHub Authentication tokens"
category: tnt
tags: ["tnt","git","github","gh auth"]
---

Github tokens are used as a safety measure to connect to repos within your github account. It also allows you to push and pull without the need to write your password or storing it anywhere. Making it a commodity and also a safety feature.

This is actually very easy, but I've already had to search for it a couple of times, so <a href="https://docs.github.com/es/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token">here it comes</a>.

**Go into GitHub > User > Settings > Developer Settings > Personal access tokens > Tokens (Classic) > Generate New Token (Classic).**

There just generate new token or renew an expired one. There are a minimum of *capablities* one must give to these tokens (The minimum required scopes are 'repo', 'workflow' and 'read:org'), which are enough for stardard push and pull in personal repos. But should not be too hard to adapt to other needs.

Then simply <a href=https://docs.github.com/es/get-started/getting-started-with-git/caching-your-github-credentials-in-git">go into the console</a> and write (you may need to [install it first](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)):

```bash
gh auth login
```

**Select Github.com; HTTPS; (First time only) Y; "Paste Authentication token"; Paste the token.** 

You're ready to go.