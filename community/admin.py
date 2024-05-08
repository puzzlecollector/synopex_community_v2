from django.contrib import admin
from .models import Question, Board, Vote, VotingOption
# Register your models here.

class QuestionAdmin(admin.ModelAdmin):
    search_fields = ['subject']

admin.site.register(Board)
admin.site.register(Question, QuestionAdmin)
admin.site.register(Vote)
admin.site.register(VotingOption)