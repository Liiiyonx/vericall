# -*- coding: utf-8 -*-
"""话术库种子与标签体系（v0.1）

三轴标签（详见 docs/话术库标注规范.md）：
  - category：诈骗类型（8 类）
  - stage：话术阶段（opening/buildup/pressure/ask）
  - risk_elements：风险要素集合

种子设计原则：骨架 + 风格轴，可复现（同 seed 同输出），
方言化留给 TTS 阶段（此处只产 dialect_style 提示标记）。
"""

CATEGORIES = {
    "impersonate_family": "冒充熟人（子女/亲友出车祸、急事借钱）",
    "impersonate_authority": "冒充公检法/政府机关（涉嫌洗钱、配合清查）",
    "fake_investment": "虚假投资理财（内部名额、高息保本）",
    "refund_cs": "冒充客服退款（快递丢失、会员扣费）",
    "lottery_prize": "中奖/积分兑换（先交税费保证金）",
    "online_loan": "网络贷款（刷流水、解冻费）",
    "elder_healthcare": "养老保健品/免费体检（专家义诊、包治百病）",
    "romance_pig": "情感杀猪盘（网恋对象引导投资）",
}

STAGES = {
    "opening": "寒暄建立身份",
    "buildup": "铺垫事由",
    "pressure": "制造紧迫/恐吓",
    "ask": "索要钱财/凭证",
}

RISK_ELEMENTS = {
    "money_transfer": "要求转账/汇款",
    "account_info": "索要银行卡号/密码",
    "verify_code": "索要短信验证码",
    "screen_share": "要求屏幕共享/远程控制",
    "face_auth": "诱导人脸识别/活体动作",
    "secrecy": "要求保密/不得告知他人",
    "time_pressure": "限定极短时间",
    "authority_threat": "冒充权威恐吓（逮捕/冻结）",
    "safe_account": "“安全账户”话术",
    "no_police": "阻止报警/核实",
}

# 类型 → 典型风险要素（生成时注入骨架，标注时校验）
CATEGORY_RISK_MAP = {
    "impersonate_family": ["money_transfer", "secrecy", "time_pressure"],
    "impersonate_authority": ["authority_threat", "safe_account", "money_transfer", "no_police"],
    "fake_investment": ["money_transfer", "time_pressure"],
    "refund_cs": ["verify_code", "screen_share", "account_info"],
    "lottery_prize": ["money_transfer", "time_pressure"],
    "online_loan": ["money_transfer", "account_info"],
    "elder_healthcare": ["money_transfer", "secrecy"],
    "romance_pig": ["money_transfer", "secrecy"],
}

# 每类 × 每阶段的骨架模板（{var} 由风格轴填充）
SEEDS = {
    ("impersonate_family", "opening"): [
        "喂{relation}，是我啊，{phone_excuse}，你别挂。",
        "{relation}吗？听得出来我是谁不？{phone_excuse}。",
    ],
    ("impersonate_family", "buildup"): [
        "{relation}，我{incident}了，现在在{place}，{matter}。",
        "我{kin}的{role}刚打电话来，说他{incident}，{matter}。",
    ],
    ("impersonate_family", "pressure"): [
        "医生说{deadline}必须交押金，晚一步就{consequence_soft}，{urgency}！",
        "对方家属就在旁边，{deadline}不赔钱就报警抓我，{urgency}！",
    ],
    ("impersonate_family", "ask"): [
        "你先转{money}到{pay_target}，{keep_secret}，回头我再跟你解释。",
        "把{money}转到我同事这个账户，名字我念给你，{no_police}。",
    ],
    ("impersonate_authority", "opening"): [
        "你好，这里是{authority}，请问是{family_name}本人吗？",
        "我是{authority}办案民警，工号{badge_no}，现在依法对你进行电话询问。",
    ],
    ("impersonate_authority", "buildup"): [
        "你{doc}涉嫌{crime}，涉案金额巨大，案件已移交我们侦办。",
        "我们查获一个{crime}团伙，其中有张银行卡登记在你名下。",
    ],
    ("impersonate_authority", "pressure"): [
        "{deadline}不配合资金清查，将{consequence}，并通知你子女单位。",
        "案件保密级别很高，{no_police}，否则按妨碍公务处理。",
    ],
    ("impersonate_authority", "ask"): [
        "为证明资金来源合法，需将存款转入{pay_target}进行核验，验完即退。",
        "下载这个'安全防护'APP，按提示做{face_action}和人脸核验。",
    ],
    ("fake_investment", "opening"): [
        "{relation}，我是老周啊，上次{benefit}见过的，还记得不？",
        "阿姨您好，我是{company}理财顾问小李，您上次咨询过我们的养老理财。",
    ],
    ("fake_investment", "buildup"): [
        "我们内部有个{opportunity}，只给老客户，{gain}，我自己都投了{money}。",
        "这个月央行有{policy}政策，名额有限，{gain}写进合同。",
    ],
    ("fake_investment", "pressure"): [
        "名额{deadline}截止，过了这个村就没这个店，{urgency}！",
        "明天系统就关闸了，您要抢到今天这单，{urgency}。",
    ],
    ("fake_investment", "ask"): [
        "您先转{money}到{pay_target}锁名额，合同下午给您送上门。",
        "开通 VIP 通道要存{money}到平台账户，随时可取，{keep_secret}。",
    ],
    ("refund_cs", "opening"): [
        "您好，我是{service}官方客服，工号{badge_no}，您有一笔订单需要处理。",
        "您好，{service}售后，系统显示您开通了自动扣费会员。",
    ],
    ("refund_cs", "buildup"): [
        "您的快递在运输中丢失，按规定双倍赔偿您{money_small}。",
        "您误开通的会员每月扣{money_small}，现在帮您取消并退款。",
    ],
    ("refund_cs", "pressure"): [
        "退款通道{deadline}关闭，逾期无法办理，{urgency}。",
        "不取消今晚十二点就会扣款{money}，{urgency}。",
    ],
    ("refund_cs", "ask"): [
        "退款需要验证本人，手机马上收到{code_name}，念给我即可到账。",
        "您打开会议软件点{screen_share_action}，我远程指导您操作退款页面。",
    ],
    ("lottery_prize", "opening"): [
        "恭喜您！您在{platform}活动中抽中{prize}，我是兑奖中心工作人员。",
        "您好，您的手机积分可兑换{prize}，今日到期。",
    ],
    ("lottery_prize", "buildup"): [
        "奖品价值{money}，只需缴纳个人所得税{money_small}即可领取。",
        "兑换需先验证身份并支付保证金{money_small}，领奖后全额退还。",
    ],
    ("lottery_prize", "pressure"): [
        "兑奖资格{deadline}作废，逾期视为自动放弃，{urgency}。",
    ],
    ("lottery_prize", "ask"): [
        "税费请转到{pay_target}，备注'个税'，转完立刻安排发货。",
        "保证金请充到平台账户，卡号我发您，{keep_secret}。",
    ],
    ("online_loan", "opening"): [
        "您好，{platform}信贷中心，您申请的{money}额度已批。",
        "需要周转资金吗？我们是正规持牌机构，秒批{money}。",
    ],
    ("online_loan", "buildup"): [
        "放款前需验证还款能力，往您自己卡里存{money_small}刷个流水。",
        "您的账户有风险标记，需缴纳{money_small}解冻费才能放款。",
    ],
    ("online_loan", "pressure"): [
        "额度{deadline}回收，今天不办下个月利率上浮，{urgency}。",
    ],
    ("online_loan", "ask"): [
        "解冻费请转到{pay_target}，放款时一并返还。",
        "流水验证需要您把验证码告诉我，系统自动核验。",
    ],
    ("elder_healthcare", "opening"): [
        "{relation}，社区组织免费体检，我给您报上名了。",
        "阿姨，我是{company}健康顾问，周四有专家义诊，专看您这个年纪的毛病。",
    ],
    ("elder_healthcare", "buildup"): [
        "专家看了您的报告，说您这情况得抓紧调理，我们有个{opportunity}。",
        "这个产品是航天员同款，{gain}，医院都不告诉你的。",
    ],
    ("elder_healthcare", "pressure"): [
        "专家只坐诊今天一天，优惠{deadline}结束，{urgency}。",
        "您这病拖不得，再拖就{consequence_soft}，{urgency}。",
    ],
    ("elder_healthcare", "ask"): [
        "一个疗程{money}，先付定金到{pay_target}，{keep_secret}，别让孩子们拦着您治病。",
        "今天交钱送理疗床，钱转到财务这个账户，{no_police}。",
    ],
    ("romance_pig", "opening"): [
        "亲爱的，在忙吗？我今天特别想你。",
        "宝贝，我刚下手术台，第一个就想跟你说话。",
    ],
    ("romance_pig", "buildup"): [
        "我最近跟导师做个{opportunity}，挺稳的，想带你赚点钱，以后咱们{future_plan}。",
        "我发现了平台的漏洞，{gain}，我自己账户给你看。",
    ],
    ("romance_pig", "pressure"): [
        "这波行情{deadline}就没了，错过要等半年，{urgency}。",
        "你要是不信我，就把钱撤出来，我什么都不说了。",
    ],
    ("romance_pig", "ask"): [
        "你先投{money}试试水，盈利了再加大，链接我发你，{keep_secret}。",
        "亲爱的，我这边保证金差{money}，你先帮我垫上，见面就还你。",
    ],
}

# 风格轴（在 generate_scam_scripts 基础上扩充）
STYLE_AXES = {
    "relation": ["妈", "爸", "奶奶", "爷爷", "阿婆", "阿公", "老李", "王阿姨", "亲家母"],
    "kin": ["儿子", "女儿", "孙子", "外孙", "老伙计"],
    "role": ["班主任", "同事", "辅导员", "主治医生", "单位领导"],
    "phone_excuse": ["手机摔坏了这是同学的号", "手机被收了我偷偷用别人手机打的", "我在外地手机丢了", "换号了这是我的新号"],
    "matter": ["我出车祸撞了人私了要赔钱", "我被骗了要还平台的钱", "手术押金不够", "人被扣在酒店了"],
    "incident": ["车祸", "突发疾病住院", "被人打了", "急性阑尾炎"],
    "place": ["医院急诊", "交警队", "外地出差的酒店"],
    "money": ["三万", "五万", "八万", "两万六", "一万八", "十二万"],
    "money_small": ["三百八", "五百", "一千二", "八百八"],
    "pay_target": ["这个卡号", "安全账户", "这个二维码", "对公账户", "财务指定账户"],
    "keep_secret": ["先别告诉我爸", "别打电话给孩子们", "这事别声张", "别跟家里人说"],
    "urgency": ["马上就要", "半小时内必须到账", "晚一步就完了", "就现在"],
    "authority": ["市公安局", "检察院", "通讯管理局", "医保局", "银保监会"],
    "doc": ["名下的银行卡", "医保账户", "身份证登记信息", "手机号"],
    "crime": ["洗钱案件", "医保诈骗", "身份盗用", "非法集资"],
    "deadline": ["今天下午五点前", "两小时内", "明天中午前", "今天二十四点前"],
    "consequence": ["冻结你全部账户并逮捕你", "案件移交法院强制执行", "上门抓捕"],
    "consequence_soft": ["耽误治疗", "落下病根", "出大问题"],
    "service": ["银行", "快递公司", "电商平台", "运营商"],
    "code_name": ["验证码", "短信校验码"],
    "screen_share_action": ["屏幕共享", "远程协助"],
    "face_action": ["眨眨眼摇摇头做人脸识别", "对着镜头念数字"],
    "benefit": ["投资说明会", "养生讲座", "免费体检"],
    "opportunity": ["内部理财名额", "原始股认购", "高息存款", "养老床位预订"],
    "gain": ["本金翻倍", "月息百分之十", "稳赚不赔", "年化百分之十五"],
    "policy": ["养老金融扶持", "数字人民币推广"],
    "company": ["康寿堂", "夕阳红健康", "中安理财"],
    "family_name": ["机主本人", "这张卡的主人"],
    "badge_no": ["0318", "5562", "0947"],
    "platform": ["幸运大转盘", "周年庆", "消费返现活动"],
    "prize": ["最新款手机", "万元现金", "按摩椅"],
    "future_plan": ["在城里买房", "一起养老", "年底就见面"],
    "no_police": ["别报警", "不许告诉别人", "千万别声张"],
}

# 方言语感标记（与红队工厂方言目录对齐：data/redteam/{mandarin,minnan,...}）
DIALECT_STYLES = [
    "mandarin-年轻男声", "mandarin-中年女声", "mandarin-老年男声",
    "minnan-带闽南腔男声", "sichuan-带川渝腔女声", "cantonese-带粤腔男声",
    "dongbei-带东北腔男声", "henan-带河南腔女声",
]
