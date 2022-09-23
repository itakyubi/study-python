# -*-coding:utf-8-*-

import requests
from lxml import etree
import time

from pip._vendor.distlib.compat import raw_input


class BaiDu_Spider(object):
    def __init__(self, keyword):
        self.base_url = 'https://www.baidu.com/s?wd={}'
        self.keyword = keyword
        self.url = self.base_url.format(self.keyword) + '&pn={}&ie=utf-8'

    def get_html(self, page):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.106 Safari/537.36'
        }
        try:
            r = requests.get(self.url.format(page), headers=headers)
            r.encoding = 'utf-8'
            res = etree.HTML(r.text)
            selector = res.xpath('//div[@id="content_left"]/div[@class="result c-container xpath-log new-pmd"]')
            data_list = []
            for data in selector:
                title = ''.join(data.xpath('./div/div/h3/a/text()'))
                link = ''.join(data.xpath('./div/div/h3/a/@href'))

                item = {
                    'title': title,
                    'link': link,
                }
                data_list.append(item)
            flag = res.xpath('//div[@id="page"]/div/a[last()]/text()')
            print(flag)
            if flag:
                return data_list, True
            else:
                return data_list, False
        except:
            pass

    def save_data(self, item):
        with open(crawl_result, 'a', encoding='utf-8')as f:
            data = item['title'] + '\t' + item['link']
            print(data)
            f.write(data + '\n')


def main():
    n = 10
    while True:
        data_list, flag = spider.get_html(n)
        for data in data_list:
            spider.save_data(data)
        time.sleep(1)
        if flag is True:
            n += 10
        else:
            print(f'程序已经退出，在{int(n / 10) + 1}页')
            break


if __name__ == '__main__':
    keyWord = raw_input('请输入需要搜索的关键词，按回车确定:')
    crawl_result = f'./crawl_{keyWord}.txt'
    spider = BaiDu_Spider(keyWord)
    main()
