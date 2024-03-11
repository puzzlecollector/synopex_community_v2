from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from ..models import Question, Answer, Comment, Board, Vote, VotingOption
from django.utils import timezone
from ..forms import QuestionForm, AnswerForm, CommentForm
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Q, Count
from django.conf import settings
from django.views.decorators.http import require_POST, require_http_methods
import requests
import pyupbit
import ccxt
import time 
import yfinance as yf
import pytz
from datetime import datetime, timedelta, timezone
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from dateutil import parser
from transformers import AutoModelForSequenceClassification, AlbertTokenizer
import torch
import torch.nn as nn
import openai
from statsmodels.tsa.arima.model import ARIMA
import pickle
import joblib
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier
import lightgbm as lgb
from common.models import PointTokenTransaction
from django.db.models import Q
import json

# it says bitget, but we are using coinbase data
# rule: korean exchange - upbit, american exchange - coinbase
def get_kimchi_data():
    coinbasepro = ccxt.coinbasepro()
    data = {}
    seoul_timezone = pytz.timezone("Asia/Seoul")
    current_time_seoul = datetime.now(seoul_timezone)
    data["current_time"] = current_time_seoul.strftime("%Y-%m-%d %H:%M:%S")
    USDKRW = yf.Ticker("USDKRW=X")
    history = USDKRW.history(period="1d")
    data["now_usd_krw"] = history["Close"].iloc[0]
    data["now_upbit_price"] = pyupbit.get_current_price("KRW-BTC")
    data["now_bitget_price"] = coinbasepro.fetch_ticker("BTC/USDT")["close"]
    data["kp"] = round((data["now_upbit_price"] * 100 / (data["now_bitget_price"] * data["now_usd_krw"])) - 100, 3)
    return data

# for coinness data scraping
def get_articles(headers, url):
    news_req = requests.get(url, headers=headers)
    soup = BeautifulSoup(news_req.content, "html.parser")
    # Extracting title
    title = soup.find("h1", {"class": "h_title"}).text.strip()
    # Finding the specific parent <div>
    article_content_div = soup.find('div', class_='par')
    content = ""  # Initialize content as empty string

    # Check if the parent div was found
    if article_content_div:
        # Extracting text from all child <div> tags within the <div class="par">, ignoring those with a class attribute
        child_divs = article_content_div.find_all('div', class_=False,
                                                  recursive=False)  # recursive=False ensures we only look at direct children
        for div in child_divs:
            # You may want to further refine how you extract text, depending on structure
            content += div.get_text(strip=True) + " "  # Appending each <div> content with a space for readability

        # Optionally, remove specific unwanted text
        unwanted_text = "이 광고는 쿠팡 파트너스 활동의 일환으로, 이에 따른 일정액의 수수료를 제공받습니다."
        content = content.replace(unwanted_text, "").strip()
    else:
        content = "No content found in the specified structure."

    return title, content

def scrape_tokenpost():
    all_titles, all_contents, all_full_times = [], [], []
    for i in tqdm(range(1, 2), desc="Scraping content from chosun"):
        try:
            links = []
            url = f"https://health.chosun.com/list.html"
            headers = {'User-Agent': 'Mozilla/5.0'}
            news_req = requests.get(url, headers=headers)
            soup = BeautifulSoup(news_req.content, "html.parser")
            elems = soup.find_all("div", {"class": "latest-list area-part"})
            for e in elems:
                article_elems = e.find_all("div", {"class": "inn"})
                for article in article_elems:
                    title_link = article.find("a", href=True)
                    full_link = "https://health.chosun.com/" + title_link['href']
                    links.append(full_link)
                    date_elem = article.find("span", {"class": "date"})
                    news_date = parser.parse(date_elem.text)
                    all_full_times.append(news_date)
                    if len(all_full_times) > 7:
                        break
            for link in links:
                try:
                    title, content = get_articles(headers, link)
                    all_titles.append(title)
                    all_contents.append(content)
                except Exception as e:
                    print(f"Error while scraping news content: {e}")

        except Exception as e:
            print(f"Error while scraping page {i}: {e}")
        time.sleep(0.1)

    if len(all_titles) != len(all_full_times):
        for k in range(len(all_full_times)):
            all_titles.append('')
            all_contents.append('')

    return pd.DataFrame({'titles': all_titles, 'contents': all_contents, 'datetimes': all_full_times})

def get_sentiment_scores(df):
    titles = df["titles"].values
    contents = df["contents"].values
    tokenizer = AlbertTokenizer.from_pretrained("aiphabtc/kr-cryptodeberta")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("aiphabtc/kr-cryptodeberta")
    scores = np.zeros(3)
    for i in range(len(titles)):
        encoded_inputs = tokenizer(str(titles[i]), str(contents[i]), max_length=512, padding="max_length",
                                   truncation=True, return_tensors="pt")
        with torch.no_grad():
            sentiment = sentiment_model(**encoded_inputs).logits
            sentiment = nn.Softmax(dim=1)(sentiment)
            sentiment = sentiment.detach().cpu().numpy()[0]
        scores += sentiment
    scores /= int(df.shape[0])
    print(scores)
    return scores  # average scores

def get_news_and_sentiment(request):
    # Your news scraping and sentiment analysis logic here
    df = scrape_tokenpost()
    scraped_data = df.to_dict(orient="records")  # converts DataFrame to list of dicts
    avg_sentiment_scores = get_sentiment_scores(df)
    avg_sentiment_scores_percentage = [round(score * 100, 2) for score in avg_sentiment_scores]
    sentiment_labels = ['호재', '악재', '중립']
    # Prepare your context data for JsonResponse
    data = {
        "scraped_data": scraped_data,
        "avg_sentiment_scores_percentage": avg_sentiment_scores_percentage,
        "sentiment_labels": sentiment_labels,
    }
    return JsonResponse(data)


def fetch_ai_technical1d(request):
    # Get the city name from the request, default to Seoul
    city = request.GET.get('city', 'Seoul')

    # Pass the city name to the get_weather_dict function
    weather_data = get_weather_dict(city)

    # Ensure weather_data is a string for formatting in the message
    weather_data_str = str(weather_data)

    message = ("다음과 같은 날씨 데이터가 주어졌을때:\n\n"
               "{}\n\n"
               "외출할때 어떤 옷을 입어야할지 주의사항은 있는지 알려줘."
               ).format(weather_data_str)
    openai.api_key = settings.OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"user", "content":message}
        ]
    )
    chat_message = response["choices"][0]["message"]["content"]
    return JsonResponse({"chat_message": chat_message})

@login_required(login_url="common:login")
@require_http_methods(["POST"])  # Ensures that this view can only be accessed with a POST request
def submit_sentiment_vote(request):
    # Check if the request is an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        option_id = request.POST.get("sentimentVoteOption")
        try:
            selected_option = VotingOption.objects.get(id=option_id)
            Vote.objects.create(vote_option=selected_option)
            # Return a success message in JSON format
            return JsonResponse({"message": "투표해주셔서 감사합니다", "status": "success"})
        except VotingOption.DoesNotExist:
            return JsonResponse({"message": "선택한 옵션이 존재하지 않습니다", "status": "error"})
    # Fallback for non-AJAX requests if necessary
    return redirect("index")

def calculate_vote_percentages(voting_options):
    total_votes = sum(option.vote_count for option in voting_options)
    if total_votes == 0:
        return [(option, 0) for option in voting_options] # avoid division by zero
    return [(option, (option.vote_count / total_votes) * 100) for option in voting_options]


def get_weather_dict(city="Seoul"):
    apikey = "097da88da3acdf1924aa9569e22f6880"
    city = city
    lang = "kr"
    units = "metric"
    api = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}&lang={lang}&units={units}"
    result = requests.get(api)
    data = json.loads(result.text)
    ret_data = {"name": data["name"],
                "weather": data["weather"][0]["description"],
                "temperature": data["main"]["temp"],
                "feels": data["main"]["feels_like"],
                "min_temp": data["main"]["temp_min"],
                "max_temp": data["main"]["temp_max"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_deg": data["wind"]["deg"],
                "wind_speed": data["wind"]["speed"]}
    return ret_data

def get_weather(city="Seoul"):
    apikey = "097da88da3acdf1924aa9569e22f6880"
    city = city
    lang = "kr"
    units = "metric"
    api = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}&lang={lang}&units={units}"
    result = requests.get(api)
    data = json.loads(result.text)
    ret_data = {"name": data["name"],
                "weather": data["weather"][0]["description"],
                "temperature": data["main"]["temp"],
                "feels": data["main"]["feels_like"],
                "min_temp": data["main"]["temp_min"],
                "max_temp": data["main"]["temp_max"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_deg": data["wind"]["deg"],
                "wind_speed": data["wind"]["speed"]}
    return JsonResponse({"weather_data":ret_data})


def index(request):
    # Define the list of board names you're interested in
    board_names = [
        'diet_logs',
        'exercise_log',
        'research_paper_sharing_board',
        'weight_disease_stress',
        'insurance_policy_information',
        'health_question_and_answers',
        'hospital_and_medicine',
        'struggle_stories'
    ]
    # Filter boards to only include those with names in the board_names list
    boards = Board.objects.filter(name__in=board_names)
    board_posts = {}
    for board in boards:
        # Fetch the top 3 posts for each board
        posts = Question.objects.filter(board=board).order_by('-create_date')[:3]
        board_posts[board] = posts

    # List of cities
    cities = ['Seoul', 'Busan', 'Incheon', 'Daegu', 'Daejeon', 'Gwangju', 'Suwon', 'Ulsan', 'Seongnam',
              'Goyang', 'Yongin', 'Cheongju', 'Jeonju', 'Changwon', 'Ansan', 'Anyang', 'Namyangju',
              'Hwaseong', 'Paju', 'Pyeongtaek']

    # Fetch weather data for the selected city or Seoul by default
    selected_city = request.GET.get('city', 'Seoul')
    weather_data = get_weather(selected_city).content
    weather_data = json.loads(weather_data.decode('utf-8'))['weather_data']

    sentiment_voting_options = VotingOption.objects.all()
    sentiment_votes = VotingOption.objects.annotate(vote_count=Count("votes")).order_by("-vote_count")
    sentiment_votes_with_percentages = calculate_vote_percentages(sentiment_votes)
    sentiment_data = {
        "labels": [option.name for option in sentiment_voting_options],
        "data": [percentage for _, percentage in sentiment_votes_with_percentages]
    }

    context = {
        "board_posts": board_posts,
        "sentiment_voting_options": sentiment_voting_options,
        "sentiment_data": sentiment_data, 
        "weather_data": weather_data,
        "cities": cities,
        "selected_city": selected_city,
    }
    return render(request, 'index.html', context)

def index_orig(request, board_name="free_board"):
    page = request.GET.get('page', '1')
    kw = request.GET.get('kw', '')
    so = request.GET.get("so", "recent")

    # Initialize the query for all questions or filter by board if board_name is given
    if board_name:
        board = get_object_or_404(Board, name=board_name)
        question_list = Question.objects.filter(board=board)
    else:
        board = None
        question_list = Question.objects.all()

    # Apply filtering based on 'so' and 'kw'
    if so == "recommend":
        question_list = question_list.annotate(num_voter=Count('voter')).order_by('-num_voter', '-create_date')
    elif so == "popular":
        question_list = question_list.annotate(num_answer=Count("answer")).order_by("-num_answer", "-create_date")
    else:
        question_list = question_list.order_by("-create_date")

    if kw:
        question_list = question_list.filter(
            Q(subject__icontains=kw) |
            Q(content__icontains=kw) |
            Q(author__username__icontains=kw) |
            Q(answer__author__username__icontains=kw)
        ).distinct()

    paginator = Paginator(question_list, 10)
    page_obj = paginator.get_page(page)
    
    context = {
        "board": board,  # Include the board in context
        "question_list": page_obj,
        'page': page,
        'kw': kw,
        'so': so
    }
    return render(request, 'aiphabtc/question_list.html', context)

def detail(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    context = {"question": question}
    return render(request, 'aiphabtc/question_detail.html', context)

def community_guideline(request):
    return render(request, "guidelines.html", {})

# for perceptive board
def get_current_price(request, ticker):
    try:
        price = pyupbit.get_current_price(ticker)
        return JsonResponse({'price': price})
    except Exception as e:
        # Handle errors or the case where the price cannot be fetched
        return JsonResponse({'error': str(e)}, status=400)

def search_results(request):
    query = request.GET.get('q')
    if query:
        questions = Question.objects.filter(Q(subject__icontains=query) | Q(content__icontains=query))
        answers = Answer.objects.filter(content__icontains=query)
        comments = Comment.objects.filter(content__icontains=query)
    else:
        questions = Answer.objects.none()
        answers = Answer.objects.none()
        comments = Comment.objects.none()

    context = {
        'query': query,
        'questions': questions,
        'answers': answers,
        'comments': comments,
    }
    return render(request, 'aiphabtc/search_results.html', context)