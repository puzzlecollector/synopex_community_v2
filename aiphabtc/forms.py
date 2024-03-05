from django import forms
from aiphabtc.models import Question, Answer, Comment

class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = ['subject', 'content']
        labels = {
            "subject": "제목",
            "content": "내용",
        }

class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ['content']
        labels = {
            'content': '답변내용',
        }
        
class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['content']
        labels = {
            'content': '댓글내용',
        }

import pyupbit
from django.utils import timezone
from datetime import timedelta
class PerceptiveBoardQuestionForm(forms.ModelForm):
    market_choices = [(ticker, ticker) for ticker in pyupbit.get_tickers(fiat="KRW")]
    market = forms.ChoiceField(choices=market_choices, label="Market")
    duration_from = forms.DateField(widget=forms.SelectDateWidget(), label="Duration From")
    duration_to = forms.DateField(widget=forms.SelectDateWidget(), label="Duration To")
    price_lower_range = forms.FloatField(label="Price Lower Range")
    price_upper_range = forms.FloatField(label="Price Upper Range")
    final_verdict = forms.ChoiceField(choices=[('bullish', 'Bullish'), ('bearish', 'Bearish')],
                                      widget=forms.RadioSelect, label="Final Verdict")

    class Meta:
        model = Question
        fields = ['subject', 'content', 'market', 'duration_from', 'duration_to', 'price_lower_range',
                  'price_upper_range', 'final_verdict']

    def __init__(self, *args, **kwargs):
        super(PerceptiveBoardQuestionForm, self).__init__(*args, **kwargs)
        today = timezone.now().date()
        # Set the default value for duration_from to today's date
        self.fields['duration_from'].initial = today
        self.fields['duration_to'].initial = today + timedelta(days=1)

    def clean(self):
        cleaned_data = super().clean()
        duration_from = cleaned_data.get("duration_from")
        duration_to = cleaned_data.get("duration_to")
        if duration_to and duration_from and duration_to < duration_from:
            self.add_error('duration_to', "Duration 'To date' must be after 'From date'.")
        return cleaned_data