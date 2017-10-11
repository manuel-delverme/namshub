import praw
import datetime

reddit = praw.Reddit(user_agent='title watcher (by /u/nonna)', client_id='dXU9dCs0vDtSxw', client_secret="kdl0bM8QiiL5YolJaEGI8ArSF-o")

# subreddit = reddit.subreddit('bitcoin')
subreddit = reddit.subreddit('all')
for submission in subreddit.stream.submissions():
    if submission is None:
        print("GOT None!")
    else:
        when = datetime.datetime.fromtimestamp(submission.created)
        print(when, submission.title)
