import requests


# -- Get dataset --
data = requests.get("https://www.scss.tcd.ie/Doug.Leith/CSU44061/week4.php").text
try:
    open("data","x")
except:
    open("data","w").write(data)
