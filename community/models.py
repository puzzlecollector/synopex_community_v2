from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Board(models.Model):
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    def __str__(self):
        return self.name

class Question(models.Model):
    board = models.ForeignKey(Board, on_delete=models.CASCADE, null=True, related_name='questions')
    author = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name="author_question")
    modify_date = models.DateTimeField(null=True, blank=True)
    subject = models.CharField(max_length=200)
    content = models.TextField()
    create_date = models.DateTimeField()
    voter = models.ManyToManyField(User, related_name="voter_question") # voter 추가
    def __str__(self):
        return self.subject 
    
class Answer(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name="author_answer")
    modify_date = models.DateTimeField(null=True, blank=True)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    content = models.TextField()
    create_date = models.DateTimeField()
    voter = models.ManyToManyField(User, related_name="voter_answer")

class Comment(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    create_date = models.DateTimeField()
    modify_date = models.DateTimeField(null=True, blank=True)
    question = models.ForeignKey(Question, null=True, blank=True, on_delete=models.CASCADE)
    answer = models.ForeignKey(Answer, null=True, blank=True, on_delete=models.CASCADE)

class VotingOption(models.Model):
    name = models.CharField(max_length=255)
    def __str__(self):
        return self.name

class Vote(models.Model):
    vote_option = models.ForeignKey(VotingOption, on_delete=models.CASCADE, related_name="votes")
    timestamp = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.vote_option.name
