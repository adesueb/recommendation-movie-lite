import datetime

def convertToTimemillis(date):
    try:
        d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").strftime('%s.%f')
        return int(float(d) * 1000)
    except:
        pass
    return 0

def checkingTimeDifferent(nowDate, beforeDate, dayThreshold):
    now = convertToTimemillis(nowDate)
    before = convertToTimemillis(beforeDate)
    return (now - before) > (dayThreshold * 86400000)
