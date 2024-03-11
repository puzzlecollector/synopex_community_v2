from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse
from ..models import Question, Answer, Comment, Board
from django.utils import timezone
from ..forms import QuestionForm, AnswerForm, CommentForm, PerceptiveBoardQuestionForm
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import pyupbit
from common.models import PointTokenTransaction

@login_required(login_url="common:login")
def question_create(request, board_name):
    board = get_object_or_404(Board, name=board_name)
    if board_name == "perceptive_board":
        form_class = PerceptiveBoardQuestionForm
    else:
        form_class = QuestionForm

    if request.method == "POST":
        form = form_class(request.POST)
        if form.is_valid():
            question = form.save(commit=False)
            question.author = request.user
            question.create_date = timezone.now()
            question.board = board

            # Handle perceptive_board specific logic
            if board_name == "perceptive_board":
                market = form.cleaned_data.get('market')
                final_verdict = form.cleaned_data.get('final_verdict')
                duration_from = form.cleaned_data.get('duration_from').strftime('%Y-%m-%d')
                duration_to = form.cleaned_data.get('duration_to').strftime('%Y-%m-%d')
                title_prefix = f"[관점][{market}][{final_verdict}][{duration_from} - {duration_to}] "
                question.subject = title_prefix + question.subject

                # Prepend the form inputs to the content
                content_prefix = f"Market: {market}\nDuration: {duration_from} to {duration_to}\nVerdict: {final_verdict}\n\n"
                question.content = content_prefix + question.content

            question.save()

            profile = request.user.profile
            score_obtained = 5
            reason = ""
            if board_name == "diet_logs":
                score_obtained = 5
                profile.score += 5
                reason = "체중 관리 팁"
            elif board_name == "exercise_log":
                score_obtained = 5
                profile.score += 5
                reason = "운동 기록 공유 게시판 게시글 작성 보상"
            elif board_name in ["research_paper_sharing_board", "insurance_policy_information", "hospital_and_medicine"]:
                score_obtained = 5
                profile.score += 3
                reason = "전문 게시판 게시글 작성 보상"
            elif board_name in ["health_question_and_answers"]:
                score_obtained = 5
                profile.score += 5
                reason = "Q&A / 리뷰 게시판 게시글 작성 보상"
            elif board_name in ["weight_disease_stress", "struggle_stories"]:
                score_obtained = 5
                profile.score += 5
                reason = "투병 일지 / 질환과 스트레스 극복 게시판 작성 보상"
            profile.save()
            PointTokenTransaction.objects.create(
                user=request.user,
                points=score_obtained,
                reason=reason
            )
            return redirect("aiphabtc:board_filtered", board_name=board.name)
    else:
        form = form_class()
    context = {"form": form, "board": board, "board_name": board_name}
    return render(request, 'aiphabtc/question_form.html', context)

@login_required(login_url="common:login")
def question_modify(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user != question.author:
        messages.error(request, "수정권한이 없습니다")
        return redirect("aiphabtc:detail", question_id=question.id)
    if request.method == "POST":
        form = QuestionForm(request.POST, instance=question)
        if form.is_valid():
            question = form.save(commit=False)
            question.author = request.user
            question.modify_date = timezone.now()
            question.save()
            return redirect("aiphabtc:detail", question_id=question.id)
    else:
        form = QuestionForm(instance=question)
    context = {"form":form}
    return render(request, "aiphabtc/question_form.html", context)

@login_required(login_url="common:login")
def question_delete(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.user != question.author:
        messages.error(request, "삭제권한이 없습니다")
        return redirect("aiphabtc:detail", question_id=question.id)
    board_name = question.board.name if question.board else None
    profile = request.user.profile
    score_obtained = -5
    reason = ""
    if board_name == "diet_logs":
        score_obtained = -5
        profile.score -= 5
        reason = "체중 관리 팁 글 삭제"
    elif board_name == "exercise_log":
        score_obtained = -5
        profile.score -= 5
        reason = "운동 기록 공유 게시판 게시글 삭제"
    elif board_name in ["research_paper_sharing_board", "insurance_policy_information", "hospital_and_medicine"]:
        score_obtained = -5
        profile.score -= 5
        reason = "전문 게시판 게시글 삭제"
    elif board_name in ["health_question_and_answers"]:
        score_obtained = -5
        profile.score -= 5
        reason = "Q&A / 리뷰 게시판 게시글 삭제"
    elif board_name in ["weight_disease_stress", "struggle_stories"]:
        score_obtained = -5
        profile.score -= 5
        reason = "투병 일지 / 질환과 스트레스 극복 게시판 삭제"
    if profile.score < 0:
        profile.score = 0
    profile.save()
    question.delete()

    PointTokenTransaction.objects.create(
        user=request.user,
        points=score_obtained,
        reason=reason
    )
    return redirect("aiphabtc:board_filtered", board_name=board_name)
