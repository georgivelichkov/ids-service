import json


def is_blacklisted(ip):
    for record in all_blacklisted_records():
        if int(record['id']) == int(ip):
            return True
    return False


def all_blacklisted_records():
    # Opening JSON file
    f = open('./data/blacklist.json')

    # returns JSON object as
    # a dictionary
    return json.load(f)
