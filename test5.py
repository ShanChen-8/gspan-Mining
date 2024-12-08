import concurrent.futures


def count_elements(sub_list):
    """
    统计子列表中元素的出现次数。
    """
    counter = {}
    for element in sub_list:
        if element in counter:
            counter[element] += 1
        else:
            counter[element] = 1
    return counter


def merge_counters(counter_list):
    """
    合并多个计数器字典。
    """
    merged_counter = {}
    for counter in counter_list:
        for key, value in counter.items():
            if key in merged_counter:
                merged_counter[key] += value
            else:
                merged_counter[key] = value
    return merged_counter


def main():
    # 创建一个列表，包含一些重复的元素
    data = [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 1, 2, 3, 4]

    # 将列表分成几个子列表
    sub_lists = [data[:5], data[5:10], data[10:15], data[15:]]

    # 使用 ProcessPoolExecutor 并行统计每个子列表中元素的出现次数
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(count_elements, sub_list): sub_list for sub_list in sub_lists}
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    # 合并所有子列表的计数结果
    final_counter = merge_counters(results)
    print('Final Counter:', final_counter)


if __name__ == "__main__":
    main()
