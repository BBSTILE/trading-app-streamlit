import streamlit as st
import sys

# === VERIFICACIÓN DE VERSIONES (SOLO PARA DEBUG) ===
st.sidebar.title("🔧 Sistema")
st.sidebar.write(f"**Python:** {sys.version.split()[0]}")
st.sidebar.write(f"**Streamlit:** {st.__version__}")

try:
    import pandas as pd
    import numpy as np
    import plotly
    import plotly.graph_objects as go
    st.sidebar.write(f"**Pandas:** {pd.__version__}")
    st.sidebar.write(f"**Numpy:** {np.__version__}")
    st.sidebar.write(f"**Plotly:** {plotly.__version__}")  # ← CORRECTO
    st.sidebar.success("✅ Todas las dependencias cargadas")
except Exception as e:
    st.sidebar.error(f"❌ Error: {e}")


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from io import BytesIO
from datetime import datetime
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, List

__version__ = "1.0.17"
__author__ = "De Crypto Club"
__email__ = "0xdcryptoclub@gmail.com"

st.set_page_config(page_title="Trading Evaluation Tester",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="expanded")

CONFIGS_DIR = "saved_configs"
if not os.path.exists(CONFIGS_DIR):
    os.makedirs(CONFIGS_DIR)

# ===== SISTEMA DE TRACKING CORREGIDO =====
@dataclass
class TradingEvent:
    event_type: str
    strategy: str
    timestamp: datetime
    amount: float = 0.0
    details: Optional[Dict] = None

# === NUEVA CLASE PARA TRACKING MENSUAL ===
class MonthlyWithdrawalTracker:
    def __init__(self):
        self.monthly_data = {}
        
    def add_withdrawal(self, date: datetime, amount: float, account_name: str, withdrawal_type: str):
        """Agrega un retiro al tracking mensual"""
        month_key = date.strftime('%Y-%m')
        
        if month_key not in self.monthly_data:
            self.monthly_data[month_key] = {
                'total': 0.0,
                'count': 0,
                'details': []
            }
        
        self.monthly_data[month_key]['total'] += amount
        self.monthly_data[month_key]['count'] += 1
        self.monthly_data[month_key]['details'].append({
            'date': date,
            'amount': amount,
            'account': account_name,
            'type': withdrawal_type
        })
    
    def get_monthly_summary(self):
        """Retorna un DataFrame con el resumen mensual"""
        if not self.monthly_data:
            return pd.DataFrame()
        
        summary_data = []
        for month, data in sorted(self.monthly_data.items()):
            summary_data.append({
                'Mes': month,
                'Total Retirado ($)': round(data['total'], 2),
                'Cantidad de Retiros': data['count'],
                'Promedio por Retiro ($)': round(data['total'] / data['count'], 2) if data['count'] > 0 else 0
            })
        
        return pd.DataFrame(summary_data)
    
    def get_detailed_transactions(self):
        """Retorna un DataFrame detallado con todos los retiros"""
        all_transactions = []
        for month, data in sorted(self.monthly_data.items()):
            for detail in data['details']:
                all_transactions.append({
                    'Fecha': detail['date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Mes': month,
                    'Cuenta': detail['account'],
                    'Tipo': detail['type'],
                    'Monto ($)': round(detail['amount'], 2)
                })
        
        return pd.DataFrame(all_transactions)

# === FIN DE NUEVA CLASE ===

class TradingTracker:
    def __init__(self):
        self.events = []
        self.summary = {
            'evaluations_started': 0,
            'evaluations_passed': 0,
            'evaluations_failed': 0,
            'masters_started': 0,
            'masters_burned': 0,
            'masters_cashed_out': 0,
            'masters_unlocked_55k': 0,
            'win_days': 0,
            'withdrawals_partial': 0,
            'withdrawals_partial_amount': 0.0,
            'withdrawals_all': 0,
            'withdrawals_all_amount': 0.0,
            'total_withdrawn': 0.0,
            'total_costs': 0.0,
            'active_accounts': [],
            'max_concurrent_active': 0
        }
        self.current_active = set()
        self.current_cashed_out = set()
        self.master_win_days = {}
        self.master_cycle_start = {}
        # === NUEVO: Añadimos el tracker mensual ===
        self.monthly_withdrawals = MonthlyWithdrawalTracker()
        
    def log_event(self, event_type: str, strategy: str, amount: float = 0.0, 
                  details: Optional[Dict] = None, timestamp: Optional[datetime] = None):
        """Registra un evento y actualiza el resumen"""
        event = TradingEvent(
            event_type=event_type,
            strategy=strategy,
            timestamp=timestamp if timestamp else datetime.now(),
            amount=amount,
            details=details
        )
        self.events.append(event)
        self.update_summary(event)
        
        # === NUEVO: Registrar retiros en el tracker mensual ===
        if event_type == 'WITHDRAW_PARTIAL':
            self.monthly_withdrawals.add_withdrawal(
                date=event.timestamp,
                amount=amount,
                account_name=strategy,
                withdrawal_type='Retiro Parcial ($1,000)'
            )
        elif event_type == 'WITHDRAW_ALL':
            self.monthly_withdrawals.add_withdrawal(
                date=event.timestamp,
                amount=amount,
                account_name=strategy,
                withdrawal_type='Gran Cashout'
            )
    
    def update_summary(self, event: TradingEvent):
        """Actualiza el resumen basado en el tipo de evento"""
        if event.event_type == 'EVAL_START':
            self.summary['evaluations_started'] += 1
        elif event.event_type == 'EVAL_PASS':
            self.summary['evaluations_passed'] += 1
            self.summary['total_costs'] += 175
        elif event.event_type == 'EVAL_FAIL':
            self.summary['evaluations_failed'] += 1
            self.summary['total_costs'] += 35
        elif event.event_type == 'MASTER_START':
            self.summary['masters_started'] += 1
            self.current_active.add(event.strategy)
            self.summary['active_accounts'] = list(self.current_active)
            if len(self.current_active) > self.summary['max_concurrent_active']:
                self.summary['max_concurrent_active'] = len(self.current_active)
        elif event.event_type == 'MASTER_BURN':
            self.summary['masters_burned'] += 1
            if event.strategy in self.current_active:
                self.current_active.remove(event.strategy)
                self.summary['active_accounts'] = list(self.current_active)
                if event.strategy in self.master_win_days:
                    del self.master_win_days[event.strategy]
            if event.strategy in self.master_cycle_start:
                del self.master_cycle_start[event.strategy]
        elif event.event_type == 'MASTER_CASHED_OUT':
            self.summary['masters_cashed_out'] += 1
            if event.strategy in self.current_active:
                self.current_active.remove(event.strategy)
                self.current_cashed_out.add(event.strategy)
                self.summary['active_accounts'] = list(self.current_active)
                if event.strategy in self.master_win_days:                
                    del self.master_win_days[event.strategy]
            if event.strategy in self.master_cycle_start:
                del self.master_cycle_start[event.strategy]
        elif event.event_type == 'MASTER_UNLOCK_55K':
            self.summary['masters_unlocked_55k'] += 1
        elif event.event_type == 'WIN_DAY':
            self.summary['win_days'] += 1
        elif event.event_type == 'WITHDRAW_PARTIAL':
            self.summary['withdrawals_partial'] += 1
            self.summary['withdrawals_partial_amount'] += event.amount
            self.summary['total_withdrawn'] += event.amount
        elif event.event_type == 'WITHDRAW_ALL':
            self.summary['withdrawals_all'] += 1
            self.summary['withdrawals_all_amount'] += event.amount
            self.summary['total_withdrawn'] += event.amount
    
    def add_trading_costs(self, amount: float):
        """Agrega comisiones de trading a los costos totales"""
        self.summary['total_costs'] += amount

    def calculate_time_metrics(self) -> Dict[str, Dict]:
        """Calcula métricas de tiempo entre eventos importantes"""
        evaluation_times = {}
        master_to_withdrawal_times = {}
        
        eval_start_times = {}
        master_start_times = {}
        
        for event in self.events:
            strategy = event.strategy
            
            if event.event_type == 'EVAL_START':
                eval_start_times[strategy] = event.timestamp
                
            elif event.event_type == 'EVAL_PASS':
                if strategy in eval_start_times:
                    duration = (event.timestamp - eval_start_times[strategy]).total_seconds()
                    evaluation_times[strategy] = duration
                    del eval_start_times[strategy]
            
            elif event.event_type == 'MASTER_START':
                master_start_times[strategy] = event.timestamp
                
            elif event.event_type == 'WITHDRAW_PARTIAL':
                if strategy in master_start_times and strategy not in master_to_withdrawal_times:
                    duration = (event.timestamp - master_start_times[strategy]).total_seconds()
                    master_to_withdrawal_times[strategy] = duration
        
        eval_stats = {'min': None, 'max': None, 'avg': None, 'count': 0}
        if evaluation_times:
            durations = list(evaluation_times.values())
            eval_stats = {
                'min': min(durations),
                'max': max(durations),
                'avg': sum(durations) / len(durations),
                'count': len(durations)
            }
        
        withdrawal_stats = {'min': None, 'max': None, 'avg': None, 'count': 0}
        if master_to_withdrawal_times:
            durations = list(master_to_withdrawal_times.values())
            withdrawal_stats = {
                'min': min(durations),
                'max': max(durations),
                'avg': sum(durations) / len(durations),
                'count': len(durations)
            }
        
        return {
            'evaluation_times': eval_stats,
            'master_to_first_withdrawal': withdrawal_stats
        }

    def get_summary_dict(self) -> Dict:
        """Retorna el resumen en formato para mostrar"""
        summary = self.summary.copy()
        summary['net_profit'] = summary['total_withdrawn'] - summary['total_costs']
        summary['active_count'] = len(self.current_active)
        summary['cashed_out_count'] = len(self.current_cashed_out)
        summary['total_masters'] = summary['masters_started']
        
        time_metrics = self.calculate_time_metrics()
        summary['evaluation_time_stats'] = time_metrics['evaluation_times']
        summary['master_to_withdrawal_stats'] = time_metrics['master_to_first_withdrawal']
        
        return summary
    
    def export_events_to_df(self) -> pd.DataFrame:
        """Exporta todos los eventos a DataFrame"""
        if not self.events:
            return pd.DataFrame()
        
        data = []
        for event in self.events:
            data.append({
                'Evento': event.event_type,
                'Estrategia': event.strategy,
                'Fecha': event.timestamp,
                'Monto': event.amount,
                'Detalles': json.dumps(event.details) if event.details else ''
            })
        return pd.DataFrame(data)
    
    def get_strategy_stats(self) -> Dict[str, Dict]:
        """Obtiene estadísticas por estrategia"""
        strategy_stats = {}
        for event in self.events:
            strat = event.strategy
            if strat not in strategy_stats:
                strategy_stats[strat] = {
                    'evaluations_started': 0,
                    'evaluations_passed': 0,
                    'evaluations_failed': 0,
                    'masters_started': 0,
                    'masters_burned': 0,
                    'masters_cashed_out': 0,
                    'withdrawals': 0,
                    'total_withdrawn': 0
                }
            
            stats = strategy_stats[strat]
            if event.event_type == 'EVAL_START':
                stats['evaluations_started'] += 1
            elif event.event_type == 'EVAL_PASS':
                stats['evaluations_passed'] += 1
            elif event.event_type == 'EVAL_FAIL':
                stats['evaluations_failed'] += 1
            elif event.event_type == 'MASTER_START':
                stats['masters_started'] += 1
            elif event.event_type == 'MASTER_BURN':
                stats['masters_burned'] += 1
            elif event.event_type == 'MASTER_CASHED_OUT':
                stats['masters_cashed_out'] += 1
            elif event.event_type in ['WITHDRAW_PARTIAL', 'WITHDRAW_ALL']:
                stats['withdrawals'] += 1
                stats['total_withdrawn'] += event.amount
        
        return strategy_stats
    
    def get_monthly_withdrawal_summary(self):
        """Retorna el resumen mensual de retiros"""
        return {
            'monthly_summary': self.monthly_withdrawals.get_monthly_summary(),
            'detailed_transactions': self.monthly_withdrawals.get_detailed_transactions()
        }

def format_duration(seconds: float) -> str:
    """Formatea segundos a una cadena legible - Versión optimizada para trading"""
    if seconds is None or seconds <= 0:
        return "N/A"
    
    # Convertir a días de trading (asumiendo días completos)
    days = seconds / (24 * 3600)
    
    # Redondear al día más cercano para simplificar
    if days < 0.5:  # Menos de medio día
        hours = seconds / 3600
        if hours < 1:
            return "< 1 hora"
        elif hours < 2:
            return "1 hora"
        else:
            return f"{int(hours)} horas"
    elif days < 1:
        return "< 1 día"
    else:
        # Redondear días
        rounded_days = round(days)
        if rounded_days == 1:
            return "1 día"
        else:
            return f"{rounded_days} días"

def get_default_config():
    return {
        "balance_inicial": 50000,
        "target_pass": 53000,
        "trailing_offset": 2500,
        "trailing_freeze_peak": 52600,
        "trailing_frozen": 50100,
        "comision_por_trade": 3.98,
        "separator": ";",
        "strategy_column": "Strategy",
        "date_column": "Entry time",
        "profit_column": "Profit",
        "date_format": "DD/MM/YYYY",
        "start_day_option": "Cada Lunes"
    }

def save_config(name, config):
    filepath = os.path.join(CONFIGS_DIR, f"{name}.json")
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    return True

def load_config(name):
    filepath = os.path.join(CONFIGS_DIR, f"{name}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

def get_saved_configs():
    if not os.path.exists(CONFIGS_DIR):
        return []
    configs = []
    for f in os.listdir(CONFIGS_DIR):
        if f.endswith(".json"):
            configs.append(f.replace(".json", ""))
    return sorted(configs)

def delete_config(name):
    filepath = os.path.join(CONFIGS_DIR, f"{name}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def parse_csv_data(uploaded_file, separator, date_column, profit_column,
                   strategy_column, date_format):
    try:
        df = pd.read_csv(uploaded_file, sep=separator, encoding='utf-8')

        if date_column not in df.columns:
            st.error(f"Columna '{date_column}' no encontrada en el archivo")
            st.write("Columnas disponibles:", df.columns.tolist())
            return None
        if profit_column not in df.columns:
            st.error(f"Columna '{profit_column}' no encontrada en el archivo")
            st.write("Columnas disponibles:", df.columns.tolist())
            return None

        try:
            if date_format == "DD/MM/YYYY":
                try:
                    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')
                except:
                    try:
                        df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                    except:
                        df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y %H:%M', errors='coerce')
            elif date_format == "MM/DD/YYYY":
                df[date_column] = pd.to_datetime(df[date_column], dayfirst=False, errors='coerce')
            else:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        except Exception as e:
            st.warning(f"Advertencia en conversión de fechas: {str(e)}")
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        if df[date_column].isna().all():
            st.error("No se pudieron parsear las fechas. Verifica el formato.")
            return None

        def parse_profit(x):
            try:
                if isinstance(x, (int, float)):
                    return float(x)
                x_str = str(x)
                x_str = x_str.replace('$', '').replace('€', '').strip()
                x_str = x_str.replace(',', '.')
                x_str = x_str.replace(' ', '')
                return float(x_str)
            except:
                return 0.0

        df[profit_column] = df[profit_column].apply(parse_profit)
        df = df.sort_values(date_column).reset_index(drop=True)
        df["DAY"] = df[date_column].dt.date

        if strategy_column not in df.columns or df[strategy_column].isna().all():
            df["Strategy"] = "Estrategia_Única"
            strategy_column = "Strategy"
        else:
            df[strategy_column] = df[strategy_column].fillna("Sin_Estrategia")

        return df, date_column, profit_column, strategy_column
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def run_evaluation_simulation(df,
                              date_column,
                              profit_column,
                              strategy_column,
                              config,
                              start_days,
                              tracker=None):
    all_results = []
    all_sheets = {}
    
    if strategy_column in df.columns:
        strategies = df[strategy_column].unique()
        st.info(f"📊 Se encontraron {len(strategies)} estrategias trabajando en conjunto")
    else:
        strategies = ["Estrategia_Única"]
    
    df_combined = df.copy()
    df_combined = df_combined.sort_values(date_column).reset_index(drop=True)

    for eval_id, start_day in enumerate(start_days, 1):
        eval_name = f"Eval_{eval_id}"
        
        if tracker:
            # Buscar la fecha real del primer trade de esta evaluación
            trades = df_combined[df_combined["DAY"] >= start_day]
            if not trades.empty:
                first_trade_date = trades.iloc[0][date_column]
                tracker.log_event('EVAL_START', eval_name, timestamp=first_trade_date)
        
        balance = config["balance_inicial"]
        peak = config["balance_inicial"]
        trailing = config["balance_inicial"] - config["trailing_offset"]
        status = "OPEN"

        records = []
        trades = df_combined[df_combined["DAY"] >= start_day]
        pass_date = None

        for _, row in trades.iterrows():
            if status != "OPEN":
                break

            pnl = row[profit_column]
            date = row[date_column]
            trade_strategy = row[strategy_column] if strategy_column in df.columns else "Estrategia_Única"

            pnl_neto = pnl - config["comision_por_trade"]
            if tracker:
                tracker.add_trading_costs(config["comision_por_trade"])

            balance += pnl_neto
            peak = max(peak, balance)

            if peak < config["trailing_freeze_peak"]:
                trailing = peak - config["trailing_offset"]
            else:
                trailing = config["trailing_frozen"]

            event = ""

            if balance <= trailing:
                status = "FAIL"
                event = "EVAL_FAIL"
                if tracker:
                    tracker.log_event('EVAL_FAIL', eval_name, timestamp=date)
            elif balance >= config["target_pass"]:
                status = "PASS"
                event = "EVAL_PASS"
                pass_date = date
                if tracker:
                    tracker.log_event('EVAL_PASS', eval_name, amount=249, timestamp=date)

            records.append({
                "Datetime": date,
                "PnL": pnl,
                "Comision": config["comision_por_trade"],
                "PnL_Neto": pnl_neto,
                "Balance": round(balance, 2),
                "Peak": round(peak, 2),
                "Trailing": round(trailing, 2),
                "Event": event,
                "Strategy": trade_strategy
            })
        
        master_records = []
        withdrawals = []
        master_status = "NOT_APPLICABLE"
        
        if status == "PASS" and pass_date:
            master_records, withdrawals, master_status = run_master_simulation(
                df_combined, date_column, profit_column, pass_date, config, tracker, eval_name
            )
        
        if master_records:
            master_sheet_name = f"{eval_name}_MASTER"
            all_sheets[master_sheet_name] = pd.DataFrame(master_records)
            
            all_results.append({
                "Estrategia": "COMBINADA",
                "Cuenta": master_sheet_name,
                "Tipo": "MASTER",
                "Fecha Inicio": pass_date.date() if pass_date else start_day,
                "Estado": master_status,
                "Balance Final": master_records[-1]["Balance"] if master_records else balance,
                "Peak Máximo": max([r["Peak"] for r in master_records]) if master_records else peak,
                "Trades": len(master_records),
                "Max Drawdown": max([r["Peak"] - r["Balance"] for r in master_records]) if master_records else 0,
                "Retiros": sum(withdrawals) if withdrawals else 0,
                "Account": 1
            })
        
        min_balance = min([r["Balance"] for r in records]) if records else balance
        max_drawdown = peak - min_balance if records else 0

        all_results.append({
            "Estrategia": "COMBINADA",
            "Cuenta": eval_name,
            "Tipo": "EVALUACIÓN",
            "Fecha Inicio": start_day,
            "Estado": status,
            "Balance Final": round(balance, 2),
            "Peak Máximo": round(peak, 2),
            "Trades": len(records),
            "Max Drawdown": round(max_drawdown, 2),
            "Account": 1
        })

        all_sheets[eval_name] = pd.DataFrame(records)

    return all_results, all_sheets

def run_master_simulation(df, date_column, profit_column, pass_date, config, tracker=None, eval_name=""):
    # === CORRECCIÓN 1: La Master debe comenzar al DÍA SIGUIENTE a las 00:00 ===
    if isinstance(pass_date, pd.Timestamp):
        pass_date_dt = pass_date
    else:
        pass_date_dt = pd.to_datetime(pass_date)
    
    next_day = pass_date_dt.date() + pd.Timedelta(days=1)
    master_start_date = pd.Timestamp(next_day).replace(hour=0, minute=0, second=0, microsecond=0)
    
    balance = 50000
    peak = balance
    trailing = peak - 2500
    withdrawals = []
    records = []

    withdrawals_done = 0
    last_withdraw_balance_ref = balance
    unlock_date = None
    win_days_after_unlock = 0
    win_days_since_cycle_start = 0
    
    if tracker:
        tracker.log_event('MASTER_START', eval_name, timestamp=master_start_date)
        tracker.master_win_days[eval_name] = {
            'unlock_date': None,
            'win_days': 0,
            'win_days_since_cycle': 0,
            'daily_pnl': {}
        }
        tracker.master_cycle_start[eval_name] = {
            'balance': balance,
            'win_days': 0
        }

    master_trades = df[df[date_column] >= master_start_date]
    
    if master_trades.empty:
        return records, withdrawals, "ACTIVE_NO_TRADES"

    daily_pnl_data = {}
    
    for _, row in master_trades.iterrows():
        pnl = row[profit_column]
        date = row[date_column]
        day = date.date()
        
        if day not in daily_pnl_data:
            daily_pnl_data[day] = 0
        daily_pnl_data[day] += pnl

    for idx, (_, row) in enumerate(master_trades.iterrows()):
        pnl = row[profit_column]
        date = row[date_column]
        day = date.date()
        
        pnl_neto = pnl - config.get("comision_por_trade", 3.98)
        if tracker:
            tracker.add_trading_costs(config.get("comision_por_trade", 3.98))

        balance += pnl_neto
        peak = max(peak, balance)

        if peak < 52600:
            trailing = peak - 2500
        else:
            trailing = 50100

        event = ""

        if balance <= trailing:
            event = "MASTER_FAIL"
            if tracker:
                tracker.log_event('MASTER_BURN', eval_name, timestamp=date)
            records.append({
                "Datetime": date,
                "PnL": pnl,
                "Comision": config.get("comision_por_trade", 3.98),
                "PnL_Neto": pnl_neto,
                "Balance": balance,
                "Peak": peak,
                "Trailing": trailing,
                "Event": event
            })
            return records, withdrawals, "FAIL"

        if unlock_date is None and balance >= 55000:
            unlock_date = date
            if tracker:
                tracker.log_event('MASTER_UNLOCK_55K', eval_name, amount=balance, timestamp=date)
                tracker.master_win_days[eval_name]['unlock_date'] = unlock_date
                tracker.master_cycle_start[eval_name] = {
                    'balance': balance,
                    'win_days': 0
                }
                win_days_after_unlock = 0
                win_days_since_cycle_start = 0

        is_last_trade_of_day = False
        if idx < len(master_trades) - 1:
            next_row = master_trades.iloc[idx + 1]
            next_day_row = next_row[date_column].date()
            if next_day_row != day:
                is_last_trade_of_day = True
        else:
            is_last_trade_of_day = True

        if is_last_trade_of_day and unlock_date is not None:
            unlock_day = unlock_date.date()
            
            if day > unlock_day:
                daily_total = daily_pnl_data.get(day, 0)
                
                if daily_total >= 150:
                    win_days_after_unlock += 1
                    win_days_since_cycle_start += 1
                    
                    if tracker:
                        tracker.master_win_days[eval_name]['win_days'] = win_days_after_unlock
                        tracker.master_win_days[eval_name]['win_days_since_cycle'] = win_days_since_cycle_start
                        tracker.master_win_days[eval_name]['daily_pnl'][day] = daily_total
                        tracker.log_event('WIN_DAY', eval_name, amount=daily_total, timestamp=date)

        withdrawal_made = False
        withdrawal_amount = 0
        
        if unlock_date is not None:
            cycle_info = tracker.master_cycle_start[eval_name] if tracker else {
                'balance': last_withdraw_balance_ref,
                'win_days': 0
            }
            
            if withdrawals_done == 0 and win_days_after_unlock >= 8:
                withdrawal_amount = 1000
                balance -= withdrawal_amount
                withdrawals.append(withdrawal_amount)
                withdrawals_done += 1
                last_withdraw_balance_ref = balance
                win_days_since_cycle_start = 0
                event = f"WITHDRAW_1"
                withdrawal_made = True
                
                if tracker:
                    tracker.log_event('WITHDRAW_PARTIAL', eval_name, amount=withdrawal_amount, timestamp=date)
                    tracker.master_cycle_start[eval_name] = {
                        'balance': balance,
                        'win_days': win_days_after_unlock
                    }
                    tracker.master_win_days[eval_name]['win_days_since_cycle'] = 0
            
            elif 1 <= withdrawals_done < 5:
                balance_increase = balance - cycle_info['balance']
                if balance_increase >= 2000 and win_days_since_cycle_start >= 8:
                    withdrawal_amount = 1000
                    balance -= withdrawal_amount
                    withdrawals.append(withdrawal_amount)
                    withdrawals_done += 1
                    last_withdraw_balance_ref = balance
                    win_days_since_cycle_start = 0
                    event = f"WITHDRAW_{withdrawals_done}"
                    withdrawal_made = True
                    
                    if tracker:
                        tracker.log_event('WITHDRAW_PARTIAL', eval_name, amount=withdrawal_amount, timestamp=date)
                        tracker.master_cycle_start[eval_name] = {
                            'balance': balance,
                            'win_days': win_days_after_unlock
                        }
                        tracker.master_win_days[eval_name]['win_days_since_cycle'] = 0
            
            elif withdrawals_done == 5:
                balance_increase = balance - cycle_info['balance']
                if balance_increase >= 2000 and win_days_since_cycle_start >= 8:
                    excess = balance - 50100
                    if excess > 0:
                        withdrawal_amount = excess
                        balance -= withdrawal_amount
                        withdrawals.append(withdrawal_amount)
                        event = "FINAL_CASHOUT"
                        withdrawal_made = True
                        
                        if tracker:
                            tracker.log_event('WITHDRAW_ALL', eval_name, amount=withdrawal_amount, timestamp=date)
                            tracker.log_event('MASTER_CASHED_OUT', eval_name, timestamp=date)
                        records.append({
                            "Datetime": date,
                            "PnL": pnl,
                            "Comision": config.get("comision_por_trade", 3.98),
                            "PnL_Neto": pnl_neto,
                            "Balance": balance,
                            "Peak": peak,
                            "Trailing": trailing,
                            "Event": event,
                            "Withdrawal": withdrawal_amount
                        })
                        return records, withdrawals, "CASHED_OUT"
        
        if withdrawal_made:
            peak = max(peak, balance)
            if peak < 52600:
                trailing = peak - 2500
            else:
                trailing = 50100

        record_entry = {
            "Datetime": date,
            "PnL": pnl,
            "Comision": config.get("comision_por_trade", 3.98),
            "PnL_Neto": pnl_neto,
            "Balance": balance,
            "Peak": peak,
            "Trailing": trailing,
            "Event": event
        }
        
        if withdrawal_made:
            record_entry["Withdrawal"] = withdrawal_amount
        
        records.append(record_entry)

    return records, withdrawals, "ACTIVE"

def calculate_metrics(eval_results):
    if not eval_results:
        return {}

    df = pd.DataFrame(eval_results)
    total_evals = len(df)
    passed = len(df[df["Estado"] == "PASS"])
    failed = len(df[df["Estado"] == "FAIL"])
    open_evals = len(df[df["Estado"] == "OPEN"])

    pass_rate = (passed / total_evals * 100) if total_evals > 0 else 0

    avg_balance = df["Balance Final"].mean()
    max_peak = df["Peak Máximo"].max()
    avg_drawdown = df["Max Drawdown"].mean()
    max_drawdown = df["Max Drawdown"].max()

    profits = df[df["Balance Final"] > 50000]["Balance Final"] - 50000
    losses = 50000 - df[df["Balance Final"] < 50000]["Balance Final"]
    profit_factor = profits.sum() / losses.sum() if losses.sum() > 0 else float('inf') if profits.sum() > 0 else 0

    return {
        "total_evals": total_evals,
        "passed": passed,
        "failed": failed,
        "open_evals": open_evals,
        "pass_rate": pass_rate,
        "avg_balance": avg_balance,
        "max_peak": max_peak,
        "avg_drawdown": avg_drawdown,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor
    }

def generate_alerts_and_recommendations(eval_results, config):
    alerts = []
    recommendations = []

    if not eval_results:
        return alerts, recommendations

    df = pd.DataFrame(eval_results)
    pass_rate = (len(df[df["Estado"] == "PASS"]) / len(df) * 100) if len(df) > 0 else 0
    avg_drawdown = df["Max Drawdown"].mean()
    max_drawdown = df["Max Drawdown"].max()

    if pass_rate < 30:
        alerts.append({
            "type": "danger",
            "title": "Tasa de éxito muy baja",
            "message": f"Solo el {pass_rate:.1f}% de las evaluaciones pasaron. Considera revisar tu estrategia."
        })
    elif pass_rate < 50:
        alerts.append({
            "type": "warning",
            "title": "Tasa de éxito por debajo del 50%",
            "message": f"El {pass_rate:.1f}% de éxito indica riesgo moderado. Revisa las condiciones de entrada."
        })
    else:
        alerts.append({
            "type": "success",
            "title": "Buena tasa de éxito",
            "message": f"El {pass_rate:.1f}% de éxito es prometedor para escalar."
        })

    trailing_distance = config["balance_inicial"] - (config["balance_inicial"] - config["trailing_offset"])
    if max_drawdown > trailing_distance * 0.8:
        alerts.append({
            "type": "danger",
            "title": "Drawdown cercano al límite",
            "message": f"El drawdown máximo (${max_drawdown:,.0f}) está muy cerca del trailing stop (${trailing_distance:,.0f})."
        })

    consecutive_fails = 0
    max_consecutive_fails = 0
    for _, row in df.iterrows():
        if row["Estado"] == "FAIL":
            consecutive_fails += 1
            max_consecutive_fails = max(max_consecutive_fails, consecutive_fails)
        else:
            consecutive_fails = 0

    if max_consecutive_fails >= 3:
        alerts.append({
            "type": "warning",
            "title": "Rachas de pérdidas detectadas",
            "message": f"Se detectaron {max_consecutive_fails} fallos consecutivos. Esto puede indicar periodos de mercado adversos."
        })

    if pass_rate >= 60:
        recommendations.append(
            "✅ Tu sistema muestra consistencia. Considera escalar gradualmente."
        )

    if avg_drawdown < trailing_distance * 0.5:
        recommendations.append(
            "✅ El drawdown promedio es bajo. Podrías considerar un trailing offset más ajustado."
        )

    if pass_rate < 50:
        recommendations.append(
            "⚠️ Revisa tus reglas de entrada y salida antes de escalar.")

    passed_evals = df[df["Estado"] == "PASS"]
    if len(passed_evals) > 0:
        avg_trades_to_pass = passed_evals["Trades"].mean()
        recommendations.append(
            f"📊 En promedio, toma {avg_trades_to_pass:.0f} trades para pasar una evaluación."
        )

    return alerts, recommendations

def create_balance_chart(eval_sheets, selected_eval):
    if selected_eval not in eval_sheets:
        return None

    df = eval_sheets[selected_eval]
    if df.empty:
        return None

    fig = make_subplots(rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Balance, Peak y Trailing",
                                        "PnL por Trade"))

    fig.add_trace(go.Scatter(x=df["Datetime"],
                             y=df["Balance"],
                             name="Balance",
                             line=dict(color="#2E86AB", width=2)),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(x=df["Datetime"],
                             y=df["Peak"],
                             name="Peak",
                             line=dict(color="#28A745", width=1, dash="dash")),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(x=df["Datetime"],
                             y=df["Trailing"],
                             name="Trailing Stop",
                             line=dict(color="#DC3545", width=1, dash="dot")),
                  row=1,
                  col=1)

    colors = ["#28A745" if pnl >= 0 else "#DC3545" for pnl in df["PnL"]]
    fig.add_trace(go.Bar(x=df["Datetime"],
                         y=df["PnL"],
                         name="PnL",
                         marker_color=colors),
                  row=2,
                  col=1)

    for _, row in df[df["Event"] != ""].iterrows():
        color = "#28A745" if row["Event"] == "EVAL_PASS" else "#DC3545"
        fig.add_vline(x=row["Datetime"], line_dash="dash", line_color=color)
        fig.add_annotation(x=row["Datetime"],
                           y=row["Balance"],
                           text=row["Event"],
                           showarrow=True,
                           arrowhead=2,
                           arrowcolor=color,
                           font=dict(color=color))

    fig.update_layout(height=600,
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1),
                      hovermode="x unified")

    return fig

def create_summary_chart(eval_results):
    if not eval_results:
        return None

    df = pd.DataFrame(eval_results)

    colors_map = {"PASS": "#28A745", "FAIL": "#DC3545", "OPEN": "#FFC107"}
    df["Color"] = df["Estado"].map(colors_map)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["Cuenta"],
            y=df["Balance Final"],
            marker_color=df["Color"],
            text=df["Estado"],
            textposition="outside",
            hovertemplate=
            "<b>%{x}</b><br>Balance: $%{y:,.2f}<br>Estado: %{text}<extra></extra>"
        ))

    fig.update_layout(title="Balance Final por Evaluación",
                      xaxis_title="Evaluación",
                      yaxis_title="Balance Final ($)",
                      height=400,
                      showlegend=False)

    return fig

def create_pass_rate_pie(eval_results):
    
    if not eval_results:
        return None

    df = pd.DataFrame(eval_results)
    status_counts = df["Estado"].value_counts()

    colors_map = {"PASS": "#28A745", "FAIL": "#DC3545", "OPEN": "#FFC107"}

    fig = go.Figure(data=[
        go.Pie(labels=status_counts.index.tolist(),
               values=status_counts.values.tolist(),
               marker_colors=[
                   colors_map.get(str(s), "#999") for s in status_counts.index
               ],
               hole=0.4,
               textinfo="label+percent",
               textposition="outside")
    ])

    fig.update_layout(title="Distribución de Resultados",
                      height=350,
                      showlegend=True)

    return fig

def create_distribution_charts(eval_results):
    if not eval_results:
        return None, None, None

    df = pd.DataFrame(eval_results)

    fig_balance = go.Figure()
    fig_balance.add_trace(
        go.Histogram(x=df["Balance Final"],
                     nbinsx=20,
                     marker_color="#2E86AB",
                     name="Balance Final"))
    fig_balance.update_layout(title="Distribución de Balance Final",
                              xaxis_title="Balance Final ($)",
                              yaxis_title="Frecuencia",
                              height=300)

    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(
        go.Histogram(x=df["Max Drawdown"],
                     nbinsx=20,
                     marker_color="#DC3545",
                     name="Max Drawdown"))
    fig_drawdown.update_layout(title="Distribución de Drawdown Máximo",
                               xaxis_title="Drawdown ($)",
                               yaxis_title="Frecuencia",
                               height=300)

    fig_trades = go.Figure()
    fig_trades.add_trace(
        go.Histogram(x=df["Trades"],
                     nbinsx=15,
                     marker_color="#28A745",
                     name="Trades"))
    fig_trades.update_layout(title="Distribución de Número de Trades",
                             xaxis_title="Número de Trades",
                             yaxis_title="Frecuencia",
                             height=300)

    return fig_balance, fig_drawdown, fig_trades

def create_risk_analysis_chart(eval_results, config):
    if not eval_results:
        return None

    df = pd.DataFrame(eval_results)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Max Drawdown"],
            y=df["Balance Final"],
            mode="markers",
            marker=dict(size=10,
                        color=[1 if s == "PASS" else 0 for s in df["Estado"]],
                        colorscale=[[0, "#DC3545"], [1, "#28A745"]],
                        showscale=False),
            text=df["Cuenta"],
            hovertemplate=
            "<b>%{text}</b><br>Drawdown: $%{x:,.2f}<br>Balance: $%{y:,.2f}<extra></extra>"
        ))

    fig.add_hline(y=config["target_pass"],
                  line_dash="dash",
                  line_color="#28A745",
                  annotation_text="Target Pass")
    fig.add_vline(x=config["trailing_offset"],
                  line_dash="dash",
                  line_color="#DC3545",
                  annotation_text="Trailing Offset")

    fig.update_layout(title="Análisis de Riesgo: Drawdown vs Balance Final",
                      xaxis_title="Drawdown Máximo ($)",
                      yaxis_title="Balance Final ($)",
                      height=400)

    return fig

def create_strategy_summary_from_sheets(eval_sheets):
    """Analiza el desempeño de cada estrategia dentro de las cuentas combinadas"""
    if not eval_sheets:
        return None
    
    strategy_stats = {}
    
    for sheet_name, sheet_df in eval_sheets.items():
        if "_MASTER" not in sheet_name and not sheet_df.empty:
            for strategy, group in sheet_df.groupby("Strategy"):
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        'total_trades': 0,
                        'total_pnl': 0.0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'max_win': float('-inf'),
                        'max_loss': float('inf'),
                        'avg_win': 0.0,
                        'avg_loss': 0.0,
                        'evaluations_participated': set()
                    }
                
                stats = strategy_stats[strategy]
                stats['total_trades'] += len(group)
                stats['total_pnl'] += group['PnL'].sum()
                stats['evaluations_participated'].add(sheet_name)
                
                wins = group[group['PnL'] > 0]
                losses = group[group['PnL'] < 0]
                
                if not wins.empty:
                    stats['winning_trades'] += len(wins)
                    stats['max_win'] = max(stats['max_win'], wins['PnL'].max())
                    if stats['winning_trades'] > 0:
                        total_wins = stats['avg_win'] * (stats['winning_trades'] - len(wins)) + wins['PnL'].sum()
                        stats['avg_win'] = total_wins / stats['winning_trades']
                
                if not losses.empty:
                    stats['losing_trades'] += len(losses)
                    stats['max_loss'] = min(stats['max_loss'], losses['PnL'].min())
                    if stats['losing_trades'] > 0:
                        total_losses = stats['avg_loss'] * (stats['losing_trades'] - len(losses)) + losses['PnL'].sum()
                        stats['avg_loss'] = total_losses / stats['losing_trades']
    
    if not strategy_stats:
        return None
    
    summary_data = []
    for strategy, stats in strategy_stats.items():
        total_trades = stats['total_trades']
        if total_trades > 0:
            win_rate = (stats['winning_trades'] / total_trades * 100)
            profit_factor = abs(stats['avg_win'] * stats['winning_trades'] / (stats['avg_loss'] * stats['losing_trades'])) if stats['losing_trades'] > 0 and stats['avg_loss'] < 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        summary_data.append({
            'Estrategia': strategy,
            'Trades Totales': total_trades,
            '% Trades Ganadores': round(win_rate, 1),
            'PnL Total': round(stats['total_pnl'], 2),
            'PnL Promedio': round(stats['total_pnl'] / total_trades, 2) if total_trades > 0 else 0,
            'Ganancia Máxima': round(stats['max_win'], 2) if stats['max_win'] != float('-inf') else 0,
            'Pérdida Máxima': round(stats['max_loss'], 2) if stats['max_loss'] != float('inf') else 0,
            'Profit Factor': round(profit_factor, 2) if profit_factor != float('inf') else '∞',
            'Participó en Evaluaciones': len(stats['evaluations_participated'])
        })
    
    return pd.DataFrame(summary_data).sort_values('PnL Total', ascending=False)

def create_tracking_summary(tracker):
    """Crea un resumen visual del tracking"""
    if not tracker or not tracker.events:
        return None, None
    
    summary = tracker.get_summary_dict()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Evaluaciones', 'Cuentas Master',
            'Retiros y Costos', 'Resultado Neto'
        ),
        specs=[[{'type': 'domain'}, {'type': 'domain'}],
               [{'type': 'bar'}, {'type': 'bar'}]],
        vertical_spacing=0.15
    )
    
    eval_labels = ['Aprobadas', 'Fallidas']
    eval_values = [summary['evaluations_passed'], summary['evaluations_failed']]
    fig.add_trace(go.Pie(
        labels=eval_labels, values=eval_values,
        marker_colors=['#28A745', '#DC3545'], hole=0.4,
        textinfo='label+percent'
    ), row=1, col=1)
    
    master_labels = ['Activas', 'Quemadas', 'Cashout']
    master_values = [
        summary['active_count'],
        summary['masters_burned'],
        summary['masters_cashed_out']
    ]
    fig.add_trace(go.Pie(
        labels=master_labels, values=master_values,
        marker_colors=['#2E86AB', '#FFC107', '#28A745'], hole=0.4,
        textinfo='label+value'
    ), row=1, col=2)
    
    retiros_costo_labels = ['Retiros Parciales', 'Gran Cashout', 'Costos']
    retiros_costo_values = [
        summary['withdrawals_partial_amount'],
        summary['withdrawals_all_amount'],
        summary['total_costs']
    ]
    colors = ['#28A745', '#17A2B8', '#DC3545']
    fig.add_trace(go.Bar(
        x=retiros_costo_labels, y=retiros_costo_values,
        marker_color=colors,
        text=[f'${v:,.0f}' for v in retiros_costo_values],
        textposition='outside'
    ), row=2, col=1)
    
    neto_labels = ['Total Retirado', 'Total Costos', 'NETO']
    neto_values = [
        summary['total_withdrawn'],
        summary['total_costs'],
        summary['net_profit']
    ]
    colors_neto = ['#28A745', '#DC3545', '#17A2B8' if summary['net_profit'] >= 0 else '#DC3545']
    fig.add_trace(go.Bar(
        x=neto_labels, y=neto_values,
        marker_color=colors_neto,
        text=[f'${v:,.0f}' for v in neto_values],
        textposition='outside'
    ), row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Resumen de Tracking - Estado General",
        title_font_size=16
    )
    
    return fig, summary

def compare_csv_files(files_data, config, start_day_option):
    comparison_results = []

    for file_name, (df, date_col, profit_col, strategy_col) in files_data.items():
        trading_days = sorted(df["DAY"].unique())

        if start_day_option == "Cada Lunes":
            start_days = [
                d for d in trading_days if pd.Timestamp(d).weekday() == 0
            ]
        elif start_day_option == "Cada día con trades":
            start_days = trading_days
        elif start_day_option == "Primer día del mes":
            start_days = []
            seen_months = set()
            for d in trading_days:
                month_key = (pd.Timestamp(d).year, pd.Timestamp(d).month)
                if month_key not in seen_months:
                    start_days.append(d)
                    seen_months.add(month_key)
        else:
            start_days = trading_days[:10]

        if start_days:
            eval_results, _ = run_evaluation_simulation(
                df, date_col, profit_col, strategy_col, config, start_days, tracker=None)
            metrics = calculate_metrics(eval_results)

            comparison_results.append({
                "Archivo": file_name,
                "Total Trades": len(df),
                "Evaluaciones": metrics.get("total_evals", 0),
                "Tasa Éxito (%)": round(metrics.get("pass_rate", 0), 1),
                "PASS": metrics.get("passed", 0),
                "FAIL": metrics.get("failed", 0),
                "Balance Promedio": round(metrics.get("avg_balance", 0), 2),
                "Drawdown Promedio": round(metrics.get("avg_drawdown", 0), 2)
            })

    return comparison_results

def export_to_excel(eval_results, eval_sheets, config=None):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Crear un resumen de comisiones
        total_comisiones = 0
        total_trades = 0
        for sheet_name, sheet in eval_sheets.items():
            if "Comision" in sheet.columns:
                total_comisiones += sheet["Comision"].sum()
                total_trades += len(sheet)
        
        # Agregar hoja de resumen financiero
        comision_por_trade = config.get("comision_por_trade", 3.98) if config else 3.98
        resumen_financiero = pd.DataFrame({
            "Métrica": ["Total Comisiones", "Comisión por Trade", "Total Trades"],
            "Valor": [total_comisiones, comision_por_trade, total_trades]
        })
        resumen_financiero.to_excel(writer, sheet_name="Resumen_Financiero", index=False)
        
        pd.DataFrame(eval_results).to_excel(writer,
                                            sheet_name="Resumen_Evals",
                                            index=False)

        for name, sheet in eval_sheets.items():
            safe_name = name[:31]
            sheet.to_excel(writer, sheet_name=safe_name, index=False)

    output.seek(0)
    return output

def calcular_resumen_estandar(eval_results, tracker):
    """Calcula el resumen estándar a partir de los resultados y el tracker"""
    if not eval_results or not tracker:
        return None
    
    summary = tracker.get_summary_dict()
    results_df = pd.DataFrame(eval_results)
    
    if "Tipo" not in results_df.columns or "Estado" not in results_df.columns:
        return None
    
    evaluaciones_df = results_df[results_df["Tipo"] == "EVALUACIÓN"]
    masters_df = results_df[results_df["Tipo"] == "MASTER"]
    
    evaluaciones_totales = len(evaluaciones_df)
    evaluaciones_aprobadas = len(evaluaciones_df[evaluaciones_df["Estado"] == "PASS"])
    evaluaciones_fallidas = len(evaluaciones_df[evaluaciones_df["Estado"] == "FAIL"])
    evaluaciones_abiertas = len(evaluaciones_df[evaluaciones_df["Estado"] == "OPEN"])
    
    masters_totales = len(masters_df)
    masters_quemadas = len(masters_df[masters_df["Estado"] == "FAIL"])
    masters_cashed_out = len(masters_df[masters_df["Estado"] == "CASHED_OUT"])
    masters_activas = len(masters_df[masters_df["Estado"] == "ACTIVE"])
    masters_sin_trades = len(masters_df[masters_df["Estado"] == "ACTIVE_NO_TRADES"])
    
    masters_activas_df = masters_df[masters_df["Estado"].isin(["ACTIVE", "ACTIVE_NO_TRADES"])]
    if not masters_activas_df.empty and "Balance Final" in masters_activas_df.columns:
        balance_promedio = masters_activas_df["Balance Final"].mean()
        balance_min = masters_activas_df["Balance Final"].min()
        balance_max = masters_activas_df["Balance Final"].max()
    else:
        balance_promedio = balance_min = balance_max = 0
    
    total_retirado = summary.get('total_withdrawn', 0)
    total_costos = summary.get('total_costs', 0)
    neto = total_retirado - total_costos
    
    resumen = {
        'evaluaciones_totales': evaluaciones_totales,
        'evaluaciones_aprobadas': evaluaciones_aprobadas,
        'evaluaciones_fallidas': evaluaciones_fallidas,
        'evaluaciones_abiertas': evaluaciones_abiertas,
        'tasa_aprobacion': (evaluaciones_aprobadas / evaluaciones_totales * 100) if evaluaciones_totales > 0 else 0,
        
        'masters_totales': masters_totales,
        'masters_quemadas': masters_quemadas,
        'masters_cashed_out': masters_cashed_out,
        'masters_activas': masters_activas,
        'masters_sin_trades': masters_sin_trades,
        'tasa_supervivencia': ((masters_totales - masters_quemadas) / masters_totales * 100) if masters_totales > 0 else 0,
        
        'balance_promedio': balance_promedio,
        'balance_min': balance_min,
        'balance_max': balance_max,
        
        'total_retirado': total_retirado,
        'total_costos': total_costos,
        'neto': neto,
        'retiros_parciales': summary.get('withdrawals_partial_amount', 0),
        'gran_cashout': summary.get('withdrawals_all_amount', 0),
        
        'tracker_evaluations_started': summary.get('evaluations_started', 0),
        'tracker_evaluations_passed': summary.get('evaluations_passed', 0),
        'tracker_evaluations_failed': summary.get('evaluations_failed', 0),
        'tracker_masters_started': summary.get('masters_started', 0),
        'tracker_masters_burned': summary.get('masters_burned', 0),
        'tracker_masters_cashed_out': summary.get('masters_cashed_out', 0),
        'tracker_active_count': summary.get('active_count', 0),
        
        'evaluation_time_stats': summary.get('evaluation_time_stats', {}),
        'master_to_withdrawal_stats': summary.get('master_to_withdrawal_stats', {})
    }
    
    return resumen

def mostrar_resumen_estandar(resumen, tracker=None):
    """Muestra el resumen estándar en la interfaz"""
    if not resumen:
        st.info("ℹ️ No hay datos para mostrar el resumen estándar")
        return
    
    st.markdown("---")
    st.header("📊 RESULTADO — RESUMEN ESTÁNDAR")
    
    if (resumen['evaluaciones_aprobadas'] != resumen['tracker_evaluations_passed'] or
        resumen['evaluaciones_fallidas'] != resumen['tracker_evaluations_failed']):
        
        st.warning("⚠️ **Nota:** Existen discrepancias menores en los cálculos entre diferentes métodos de conteo.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧪 Evaluaciones")
        st.metric("Evaluaciones Totales", resumen['evaluaciones_totales'])
        st.metric("Evaluaciones Aprobadas", resumen['evaluaciones_aprobadas'], 
                 f"{resumen['tasa_aprobacion']:.1f}%")
        st.metric("Evaluaciones Fallidas", resumen['evaluaciones_fallidas'])
        if resumen['evaluaciones_abiertas'] > 0:
            st.metric("Evaluaciones Abiertas", resumen['evaluaciones_abiertas'])
        
        st.markdown("---")
        st.markdown("### 🧠 Másters")
        st.metric("Másters Totales", resumen['masters_totales'])
        st.metric("Másters Activas", resumen['masters_activas'])
        st.metric("Másters Quemadas", resumen['masters_quemadas'])
        if resumen['masters_cashed_out'] > 0:
            st.metric("Cashouts Completados", resumen['masters_cashed_out'])
    
    with col2:
        st.markdown("### 💰 Finanzas")
        st.metric("Total Retirado", f"${resumen['total_retirado']:,.0f}")
        st.metric("Total Costos", f"${resumen['total_costos']:,.0f}")
        
        neto_color = "normal" if resumen['neto'] >= 0 else "inverse"
        st.metric("Resultado Neto", f"${resumen['neto']:,.0f}", 
                 delta_color=neto_color)
        
        st.markdown("---")
        st.markdown("### 📊 Detalle Retiros")
        st.markdown(f"**Retiros parciales:** ${resumen['retiros_parciales']:,.0f}")
        st.markdown(f"**Gran cashout:** ${resumen['gran_cashout']:,.0f}")
        
        st.markdown("---")
        st.markdown("### 📦 Estado Actual")
        if resumen['masters_activas'] > 0:
            st.success(f"✅ **{resumen['masters_activas']} máster(s) activa(s)**")
            st.markdown(f"*Balance promedio: ${resumen['balance_promedio']:,.0f}*")
            st.markdown(f"*Rango: ${resumen['balance_min']:,.0f} - ${resumen['balance_max']:,.0f}*")
        else:
            st.warning("⚠️ **Sin másters activas**")
    
    st.markdown("---")
    st.markdown("### ⏱️ Métricas de Tiempo")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("**📊 Evaluación → Master**")
        st.markdown("*Tiempo desde inicio evaluación hasta aprobación*")
        
        eval_stats = resumen.get('evaluation_time_stats', {})
        if eval_stats and eval_stats.get('count', 0) > 0:
            st.markdown(f"• **Mínimo:** {format_duration(eval_stats['min'])}")
            st.markdown(f"• **Máximo:** {format_duration(eval_stats['max'])}")
            st.markdown(f"• **Promedio:** {format_duration(eval_stats['avg'])}")
            st.markdown(f"• **Total evaluaciones:** {eval_stats['count']}")
        else:
            st.info("No hay datos de tiempo de evaluación")
    
    with col6:
        st.markdown("**🏆 Master → Primer Retiro**")
        st.markdown("*Tiempo desde que se convierte en Master hasta el primer retiro*")
        
        with_stats = resumen.get('master_to_withdrawal_stats', {})
        if with_stats and with_stats.get('count', 0) > 0:
            st.markdown(f"• **Mínimo:** {format_duration(with_stats['min'])}")
            st.markdown(f"• **Máximo:** {format_duration(with_stats['max'])}")
            st.markdown(f"• **Promedio:** {format_duration(with_stats['avg'])}")
            st.markdown(f"• **Total masters con retiro:** {with_stats['count']}")
        else:
            st.info("No hay datos de tiempo hasta primer retiro")
    
    # === NUEVA SECCIÓN: RETIROS POR MES ===
    if tracker and hasattr(tracker, 'monthly_withdrawals'):
        st.markdown("---")
        st.header("💰 RETIROS POR MES")
        
        withdrawal_data = tracker.get_monthly_withdrawal_summary()
        
        if not withdrawal_data['monthly_summary'].empty:
            st.subheader("📅 Resumen Mensual")
            
            col7, col8, col9 = st.columns(3)
            
            with col7:
                total_meses = len(withdrawal_data['monthly_summary'])
                st.metric("Meses con Retiros", total_meses)
            
            with col8:
                total_retirado = withdrawal_data['monthly_summary']['Total Retirado ($)'].sum()
                st.metric("Total Retirado", f"${total_retirado:,.0f}")
            
            with col9:
                total_retiros = withdrawal_data['monthly_summary']['Cantidad de Retiros'].sum()
                st.metric("Total Retiros", total_retiros)
            
            st.dataframe(withdrawal_data['monthly_summary'], use_container_width=True)
            
            # Gráfico de retiros mensuales
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=withdrawal_data['monthly_summary']['Mes'],
                y=withdrawal_data['monthly_summary']['Total Retirado ($)'],
                name='Total Retirado',
                marker_color='#28A745',
                text=[f'${v:,.0f}' for v in withdrawal_data['monthly_summary']['Total Retirado ($)']],
                textposition='outside'
            ))
            fig.update_layout(
                title='Total Retirado por Mes',
                xaxis_title='Mes',
                yaxis_title='Monto ($)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detalle de transacciones
            with st.expander("📋 Ver Detalle de Todos los Retiros"):
                if not withdrawal_data['detailed_transactions'].empty:
                    st.dataframe(withdrawal_data['detailed_transactions'], use_container_width=True)
                    
                    # Botón para descargar
                    csv_data = withdrawal_data['detailed_transactions'].to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Detalle de Retiros",
                        data=csv_data,
                        file_name=f"retiros_mensuales_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.info("ℹ️ No hay datos de retiros mensuales")
    else:
        st.markdown("---")
        st.info("ℹ️ Ejecuta una simulación para ver los retiros por mes")
    
    st.markdown("---")
    st.markdown("### 📈 Análisis Final")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Resumen de Evaluaciones:**")
        if resumen['evaluaciones_totales'] > 0:
            st.markdown(f"- Tasa de aprobación: **{resumen['tasa_aprobacion']:.1f}%**")
            st.markdown(f"- Total evaluaciones: **{resumen['evaluaciones_totales']}**")
        
        if resumen['masters_totales'] > 0:
            st.markdown(f"- Supervivencia másters: **{resumen['tasa_supervivencia']:.1f}%**")
            st.markdown(f"- Másters iniciadas: **{resumen['masters_totales']}**")

        if tracker and tracker.master_win_days:
            st.markdown("**Win-Days por Máster:**")
            for master_name, data in tracker.master_win_days.items():
                if data['unlock_date']:
                    st.markdown(f"- {master_name}: {data['win_days']}/8 días")
    
    with col4:
        st.markdown("**Resultado Financiero:**")
        if resumen['neto'] > 0:
            st.markdown(f"- **✅ GANANCIA NETA: ${resumen['neto']:,.0f}**")
            roi = (resumen['neto'] / resumen['total_costos'] * 100) if resumen['total_costos'] > 0 else float('inf')
            if roi != float('inf'):
                st.markdown(f"- ROI: **{roi:.1f}%**")
            else:
                st.markdown("- ROI: **∞**")
        else:
            st.markdown(f"- **❌ PÉRDIDA NETA: ${abs(resumen['neto']):,.0f}**")
        
        if resumen['total_retirado'] > 0:
            st.markdown(f"- Total retirado: **${resumen['total_retirado']:,.0f}**")

st.title("📊 Trading Evaluation Tester")
st.markdown("Herramienta avanzada para testear tu sistema de trading y analizar su escalamiento")

tabs = st.tabs(["🏠 Evaluación Principal", "📊 Comparar Archivos", "📈 Análisis Avanzado"])

def apply_config_to_session(config_data):
    for key, value in config_data.items():
        st.session_state[key] = value

if "config_initialized" not in st.session_state:
    st.session_state["config_initialized"] = True
    defaults = get_default_config()
    apply_config_to_session(defaults)

with st.sidebar:
    st.header("⚙️ Configuración")

    with st.expander("💾 Guardar/Cargar Configuración", expanded=False):
        saved_configs = get_saved_configs()

        if saved_configs:
            selected_config = st.selectbox("Configuraciones guardadas",
                                           ["-- Seleccionar --"] + saved_configs,
                                           key="config_selector")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Cargar", use_container_width=True):
                    if selected_config != "-- Seleccionar --":
                        loaded = load_config(selected_config)
                        if loaded:
                            apply_config_to_session(loaded)
                            if "eval_results" in st.session_state:
                                del st.session_state["eval_results"]
                            if "eval_sheets" in st.session_state:
                                del st.session_state["eval_sheets"]
                            if "tracker" in st.session_state:
                                del st.session_state["tracker"]
                            st.success(f"Configuración '{selected_config}' cargada")
                            st.rerun()
            with col2:
                if st.button("🗑️ Eliminar", use_container_width=True):
                    if selected_config != "-- Seleccionar --":
                        delete_config(selected_config)
                        st.success(f"Configuración '{selected_config}' eliminada")
                        st.rerun()

        st.markdown("---")
        new_config_name = st.text_input("Nombre para guardar", placeholder="Mi estrategia")
        if st.button("💾 Guardar configuración actual", use_container_width=True):
            if new_config_name:
                current_config = {
                    "balance_inicial": st.session_state.get("balance_inicial", 50000),
                    "target_pass": st.session_state.get("target_pass", 53000),
                    "trailing_offset": st.session_state.get("trailing_offset", 2500),
                    "trailing_freeze_peak": st.session_state.get("trailing_freeze_peak", 52600),
                    "trailing_frozen": st.session_state.get("trailing_frozen", 50100),
                    "separator": st.session_state.get("separator", ";"),
                    "date_column": st.session_state.get("date_column", "Entry time"),
                    "profit_column": st.session_state.get("profit_column", "Profit"),
                    "strategy_column": st.session_state.get("strategy_column", "Strategy"),
                    "date_format": st.session_state.get("date_format", "DD/MM/YYYY"),
                    "start_day_option": st.session_state.get("start_day_option", "Cada Lunes"),
                    "comision_por_trade": st.session_state.get("comision_por_trade", 3.98)
                }
                save_config(new_config_name, current_config)
                st.success(f"Configuración '{new_config_name}' guardada")
                st.rerun()

    st.subheader("📁 Archivo de Datos")
    uploaded_file = st.file_uploader("Cargar archivo CSV de NinjaTrader", type=["csv"],
                                     help="Archivo CSV exportado de NinjaTrader con datos de trades")

    st.subheader("📋 Formato del CSV")
    separator_options = [";", ",", "\t"]
    separator = st.selectbox("Separador", separator_options, key="separator")
    date_column = st.text_input("Columna de Fecha", key="date_column")
    profit_column = st.text_input("Columna de Profit", key="profit_column")
    strategy_column = st.text_input("Columna de Estrategia", value="Strategy", key="strategy_column")
    date_format_options = ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"]
    date_format = st.selectbox("Formato de Fecha", date_format_options, key="date_format")

    st.subheader("💰 Parámetros de Evaluación")
    balance_inicial = st.number_input("Balance Inicial ($)", step=1000, min_value=1000, key="balance_inicial")
    target_pass = st.number_input("Target para Pasar ($)", step=500, min_value=1000, key="target_pass")
    trailing_offset = st.number_input("Trailing Offset ($)", step=100, min_value=100, key="trailing_offset")
    trailing_freeze_peak = st.number_input("Trailing Freeze Peak ($)", step=100, key="trailing_freeze_peak")
    trailing_frozen = st.number_input("Trailing Frozen ($)", step=100, key="trailing_frozen")
    comision_por_trade = st.number_input("Comisión por Trade ($)", step=0.01, min_value=0.0, value=3.98, key="comision_por_trade")

    st.subheader("📅 Días de Inicio")
    start_day_options = ["Cada Lunes", "Cada día con trades", "Primer día del mes", "Personalizado"]
    start_day_option = st.selectbox("Iniciar evaluaciones desde", start_day_options, key="start_day_option")

config = {
    "balance_inicial": balance_inicial,
    "target_pass": target_pass,
    "trailing_offset": trailing_offset,
    "trailing_freeze_peak": trailing_freeze_peak,
    "trailing_frozen": trailing_frozen,
    "master_balance": 50000,
    "master_trailing": 2500,
    "master_unlock": 55000,
    "comision_por_trade": comision_por_trade
}

with tabs[0]:
    if uploaded_file is not None:
        current_file_name = uploaded_file.name
        if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != current_file_name:
            st.session_state["last_uploaded_file"] = current_file_name
            if "eval_results" in st.session_state:
                del st.session_state["eval_results"]
            if "eval_sheets" in st.session_state:
                del st.session_state["eval_sheets"]
            if "tracker" in st.session_state:
                del st.session_state["tracker"]

        result = parse_csv_data(uploaded_file, separator, date_column, profit_column, strategy_column, date_format)

        if result is not None:
            df, date_col, profit_col, strategy_col = result
            st.success(f"✅ Archivo cargado correctamente: {len(df)} trades encontrados")

            with st.expander("👁️ Vista previa de datos", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)

            trading_days = sorted(df["DAY"].unique())

            if start_day_option == "Cada Lunes":
                start_days = [d for d in trading_days if pd.Timestamp(d).weekday() == 0]
            elif start_day_option == "Cada día con trades":
                start_days = trading_days
            elif start_day_option == "Primer día del mes":
                start_days = []
                seen_months = set()
                for d in trading_days:
                    month_key = (pd.Timestamp(d).year, pd.Timestamp(d).month)
                    if month_key not in seen_months:
                        start_days.append(d)
                        seen_months.add(month_key)
            else:
                selected_dates = st.multiselect("Seleccionar días de inicio", options=trading_days,
                                               default=trading_days[:5] if len(trading_days) >= 5 else trading_days)
                start_days = selected_dates

            if start_days:
                total_evals = len(start_days)
                st.info(f"📅 Se simularán {total_evals} evaluaciones")

                if st.button("🚀 Ejecutar Simulación", type="primary", use_container_width=True):
                    with st.spinner("Ejecutando simulación..."):
                        tracker = TradingTracker()
                        eval_results, eval_sheets = run_evaluation_simulation(
                            df, date_col, profit_col, strategy_col, config, start_days, tracker
                        )
                        st.session_state["eval_results"] = eval_results
                        st.session_state["eval_sheets"] = eval_sheets
                        st.session_state["tracker"] = tracker
                        st.success("✅ Simulación completada")
                        st.rerun()
            else:
                st.warning("⚠️ No hay días de inicio seleccionados")

        if "eval_results" in st.session_state and st.session_state["eval_results"]:
            eval_results = st.session_state["eval_results"]
            eval_sheets = st.session_state["eval_sheets"]

            st.markdown("---")
            st.header("📈 Dashboard de Resultados")
            metrics = calculate_metrics(eval_results)

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Evaluaciones", metrics["total_evals"])
            with col2:
                delta_color = "normal" if metrics["pass_rate"] >= 50 else "inverse"
                st.metric("Tasa de Éxito", f"{metrics['pass_rate']:.1f}%",
                         delta=f"{metrics['passed']} PASS / {metrics['failed']} FAIL", delta_color=delta_color)
            with col3:
                st.metric("Balance Promedio", f"${metrics['avg_balance']:,.2f}")
            with col4:
                st.metric("Drawdown Máximo", f"${metrics['max_drawdown']:,.2f}")
            with col5:
                pf_display = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "∞"
                st.metric("Profit Factor", pf_display)

            st.markdown("---")
            st.header("🔄 Análisis por Estrategia (Dentro de Cuentas Combinadas)")
            strategy_summary = create_strategy_summary_from_sheets(eval_sheets)
            
            if strategy_summary is not None and len(strategy_summary) > 0:
                st.dataframe(strategy_summary, use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=strategy_summary["Estrategia"], y=strategy_summary["PnL Total"],
                                     name="PnL Total", marker_color="#28A745",
                                     text=[f"${v:,.0f}" for v in strategy_summary["PnL Total"]],
                                     textposition="outside"))
                fig.update_layout(title="Contribución de PnL por Estrategia", xaxis_title="Estrategia",
                                yaxis_title="PnL Total ($)", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=strategy_summary["Estrategia"], y=strategy_summary["% Trades Ganadores"],
                                      name="Win Rate %", marker_color="#2E86AB",
                                      text=[f"{v}%" for v in strategy_summary["% Trades Ganadores"]],
                                      textposition="outside"))
                fig2.update_layout(title="Tasa de Aciertos por Estrategia", xaxis_title="Estrategia",
                                 yaxis_title="% Trades Ganadores", height=400)
                st.plotly_chart(fig2, use_container_width=True)
                
                selected_strategy = st.selectbox("Ver detalles de estrategia específica:",
                                                options=strategy_summary["Estrategia"].tolist())
                
                if selected_strategy:
                    strategy_trades = []
                    for sheet_name, sheet_df in eval_sheets.items():
                        if "_MASTER" not in sheet_name:
                            strat_trades = sheet_df[sheet_df["Strategy"] == selected_strategy]
                            if not strat_trades.empty:
                                strategy_trades.append(strat_trades)
                    
                    if strategy_trades:
                        all_strat_trades = pd.concat(strategy_trades)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"Trades Totales", len(all_strat_trades))
                        with col2:
                            win_trades = len(all_strat_trades[all_strat_trades["PnL"] > 0])
                            st.metric(f"Trades Ganadores", win_trades, f"{win_trades/len(all_strat_trades)*100:.1f}%")
                        with col3:
                            st.metric(f"PnL Total", f"${all_strat_trades['PnL'].sum():,.2f}")
                        
                        fig3 = go.Figure()
                        fig3.add_trace(go.Histogram(x=all_strat_trades["PnL"], nbinsx=30,
                                                   marker_color=["#28A745" if x > 0 else "#DC3545" for x in all_strat_trades["PnL"]],
                                                   name="Distribución PnL"))
                        fig3.update_layout(title=f"Distribución de PnL - {selected_strategy}",
                                         xaxis_title="PnL ($)", yaxis_title="Frecuencia", height=300)
                        st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("ℹ️ Solo se encontró una estrategia en los datos.")

            st.markdown("---")
            st.header("📊 Resumen de Tracking - Eventos y Finanzas")

            if "tracker" in st.session_state:
                tracker = st.session_state["tracker"]
                tracking_chart, summary = create_tracking_summary(tracker)
                if tracking_chart:
                    st.plotly_chart(tracking_chart, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Evaluaciones Aprobadas", summary['evaluations_passed'],
                             f"Fallidas: {summary['evaluations_failed']}")
                with col2:
                    st.metric("Cuentas Master", f"{summary['masters_started']}",
                             f"Activas: {summary['active_count']} | Quemadas: {summary['masters_burned']}")
                with col3:
                    st.metric("Total Retirado", f"${summary['total_withdrawn']:,.0f}",
                             f"Parciales: ${summary['withdrawals_partial_amount']:,.0f}")
                with col4:
                    neto_color = "normal" if summary['net_profit'] >= 0 else "inverse"
                    st.metric("Resultado Neto", f"${summary['net_profit']:,.0f}",
                             f"Costos: ${summary['total_costs']:,.0f}", delta_color=neto_color)
                
                with st.expander("📋 Ver Todos los Eventos Registrados", expanded=False):
                    events_df = tracker.export_events_to_df()
                    if not events_df.empty:
                        st.dataframe(events_df, use_container_width=True)
                        csv_events = events_df.to_csv(index=False)
                        st.download_button(label="📥 Descargar Eventos CSV", data=csv_events,
                                         file_name=f"eventos_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                         mime="text/csv", use_container_width=True)
                
                with st.expander("📊 Estadísticas por Estrategia", expanded=False):
                    strategy_stats = tracker.get_strategy_stats()
                    if strategy_stats:
                        stats_data = []
                        for strat, stats in strategy_stats.items():
                            stats_data.append({
                                'Estrategia': strat,
                                'Eval Iniciadas': stats['evaluations_started'],
                                'Eval Aprobadas': stats['evaluations_passed'],
                                'Eval Fallidas': stats['evaluations_failed'],
                                'Masters Iniciadas': stats['masters_started'],
                                'Masters Quemadas': stats['masters_burned'],
                                'Total Retirado': stats['total_withdrawn']
                            })
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("ℹ️ Ejecuta una simulación para ver el resumen de tracking")

            alerts, recommendations = generate_alerts_and_recommendations(eval_results, config)
            if alerts:
                st.markdown("### ⚠️ Alertas y Recomendaciones")
                for alert in alerts:
                    if alert["type"] == "danger":
                        st.error(f"**{alert['title']}**: {alert['message']}")
                    elif alert["type"] == "warning":
                        st.warning(f"**{alert['title']}**: {alert['message']}")
                    else:
                        st.success(f"**{alert['title']}**: {alert['message']}")

                if recommendations:
                    with st.expander("📋 Recomendaciones detalladas", expanded=False):
                        for rec in recommendations:
                            st.markdown(f"- {rec}")

            col_chart1, col_chart2 = st.columns([2, 1])
            with col_chart1:
                summary_chart = create_summary_chart(eval_results)
                if summary_chart:
                    st.plotly_chart(summary_chart, use_container_width=True)
            with col_chart2:
                pie_chart = create_pass_rate_pie(eval_results)
                if pie_chart:
                    st.plotly_chart(pie_chart, use_container_width=True)

            st.markdown("---")
            st.header("🔍 Detalle por Evaluación")
            eval_names = list(eval_sheets.keys())
            selected_eval = st.selectbox("Seleccionar Evaluación", eval_names)

            if selected_eval:
                eval_info = next((e for e in eval_results if e["Cuenta"] == selected_eval), None)
                if eval_info:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        status_color = "🟢" if eval_info["Estado"] == "PASS" else ("🔴" if eval_info["Estado"] == "FAIL" else "🟡")
                        st.metric("Estado", f"{status_color} {eval_info['Estado']}")
                    with col2:
                        st.metric("Balance Final", f"${eval_info['Balance Final']:,.2f}")
                    with col3:
                        st.metric("Peak Máximo", f"${eval_info['Peak Máximo']:,.2f}")
                    with col4:
                        st.metric("Total Trades", eval_info["Trades"])

                balance_chart = create_balance_chart(eval_sheets, selected_eval)
                if balance_chart:
                    st.plotly_chart(balance_chart, use_container_width=True)

                with st.expander("📊 Tabla de Trades", expanded=False):
                    st.dataframe(eval_sheets[selected_eval], use_container_width=True)

            st.markdown("---")
            st.header("📋 Resumen de Todas las Evaluaciones")
            results_df = pd.DataFrame(eval_results)

            def highlight_status(val):
                if val == "PASS":
                    return "background-color: #d4edda; color: #155724"
                elif val == "FAIL":
                    return "background-color: #f8d7da; color: #721c24"
                return "background-color: #fff3cd; color: #856404"

            styled_df = results_df.style.map(highlight_status, subset=["Estado"])
            st.dataframe(styled_df, use_container_width=True)

            st.markdown("---")
            st.header("📥 Exportar Resultados")
            col1, col2 = st.columns(2)
            with col1:
                excel_data = export_to_excel(eval_results, eval_sheets, config)
                st.download_button(label="📥 Descargar Excel Completo", data=excel_data,
                                 file_name=f"Evaluaciones_Trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 use_container_width=True)
            with col2:
                csv_data = results_df.to_csv(index=False)
                st.download_button(label="📥 Descargar Resumen CSV", data=csv_data,
                                 file_name=f"Resumen_Evaluaciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                 mime="text/csv", use_container_width=True)

            st.markdown("---")
            st.header("📊 RESULTADO — RESUMEN ESTÁNDAR")
            has_tracker = "tracker" in st.session_state and st.session_state["tracker"] is not None
            has_eval_results = "eval_results" in st.session_state and st.session_state["eval_results"]
            
            if has_tracker and has_eval_results:
                tracker = st.session_state["tracker"]
                eval_results = st.session_state["eval_results"]
                resumen = calcular_resumen_estandar(eval_results, tracker)
                
                if resumen:
                    mostrar_resumen_estandar(resumen, tracker)
                else:
                    st.info("ℹ️ No se pudo calcular el resumen estándar")
            else:
                st.info("ℹ️ Ejecuta una simulación para ver el resumen estándar")

    else:
        st.markdown("---")
        st.info("👆 Carga un archivo CSV de NinjaTrader en la barra lateral para comenzar")
        st.markdown("### 📖 Cómo usar esta herramienta")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **1. Preparar el archivo CSV**
            - Exporta tus trades desde NinjaTrader
            - Asegúrate de incluir columnas de fecha y profit
            """)
        with col2:
            st.markdown("""
            **2. Configurar parámetros**
            - Ajusta el balance inicial
            - Define tu target de ganancia
            - Configura los trailing stops
            """)
        st.markdown("---")
        st.markdown("### 📊 Parámetros por defecto")
        params_df = pd.DataFrame({
            "Parámetro": ["Balance Inicial", "Target Pass", "Trailing Offset", "Trailing Freeze Peak", "Trailing Frozen"],
            "Valor": ["$50,000", "$53,000", "$2,500", "$52,600", "$50,100"],
            "Descripción": [
                "Capital inicial de la cuenta",
                "Balance necesario para aprobar la evaluación",
                "Distancia del trailing stop desde el peak",
                "Peak a partir del cual el trailing se congela",
                "Nivel fijo del trailing cuando está congelado"
            ]
        })
        st.table(params_df)

with tabs[1]:
    st.header("📊 Comparar Múltiples Archivos CSV")
    st.markdown("Sube varios archivos CSV para comparar el rendimiento de diferentes periodos o estrategias")
    comparison_files = st.file_uploader("Cargar archivos CSV para comparar", type=["csv"],
                                       accept_multiple_files=True, key="comparison_files")

    if comparison_files and len(comparison_files) >= 2:
        files_data = {}
        for file in comparison_files:
            result = parse_csv_data(file, separator, date_column, profit_column, strategy_column, date_format)
            if result:
                files_data[file.name] = result

        if len(files_data) >= 2:
            st.success(f"✅ {len(files_data)} archivos cargados correctamente")
            if st.button("🔄 Comparar Archivos", type="primary", use_container_width=True):
                with st.spinner("Comparando archivos..."):
                    comparison_results = compare_csv_files(files_data, config, start_day_option)
                    st.session_state["comparison_results"] = comparison_results

            if "comparison_results" in st.session_state and st.session_state["comparison_results"]:
                comparison_df = pd.DataFrame(st.session_state["comparison_results"])
                st.markdown("### 📈 Resultados de Comparación")
                st.dataframe(comparison_df, use_container_width=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(name="Tasa de Éxito (%)", x=comparison_df["Archivo"],
                                   y=comparison_df["Tasa Éxito (%)"], marker_color="#28A745"))
                fig.update_layout(title="Comparación de Tasa de Éxito por Archivo",
                                xaxis_title="Archivo", yaxis_title="Tasa de Éxito (%)", height=400)
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name="Balance Promedio", x=comparison_df["Archivo"],
                                    y=comparison_df["Balance Promedio"], marker_color="#2E86AB"))
                fig2.add_trace(go.Bar(name="Drawdown Promedio", x=comparison_df["Archivo"],
                                    y=comparison_df["Drawdown Promedio"], marker_color="#DC3545"))
                fig2.update_layout(title="Balance y Drawdown Promedio por Archivo",
                                 xaxis_title="Archivo", yaxis_title="Monto ($)",
                                 barmode="group", height=400)
                st.plotly_chart(fig2, use_container_width=True)
    elif comparison_files:
        st.warning("⚠️ Necesitas subir al menos 2 archivos para comparar")
    else:
        st.info("👆 Sube al menos 2 archivos CSV para comparar su rendimiento")

with tabs[2]:
    st.header("📈 Análisis Avanzado de Riesgo")
    if "eval_results" in st.session_state and st.session_state["eval_results"]:
        eval_results = st.session_state["eval_results"]
        st.markdown("### 📊 Distribución de Resultados")
        fig_balance, fig_drawdown, fig_trades = create_distribution_charts(eval_results)
        col1, col2, col3 = st.columns(3)
        with col1:
            if fig_balance:
                st.plotly_chart(fig_balance, use_container_width=True)
        with col2:
            if fig_drawdown:
                st.plotly_chart(fig_drawdown, use_container_width=True)
        with col3:
            if fig_trades:
                st.plotly_chart(fig_trades, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🎯 Análisis de Riesgo")
        risk_chart = create_risk_analysis_chart(eval_results, config)
        if risk_chart:
            st.plotly_chart(risk_chart, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Estadísticas Detalladas")
        df = pd.DataFrame(eval_results)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Estadísticas de Balance Final**")
            balance_stats = df["Balance Final"].describe()
            stats_df = pd.DataFrame({
                "Métrica": ["Mínimo", "Máximo", "Promedio", "Mediana", "Desv. Estándar"],
                "Valor": [
                    f"${balance_stats['min']:,.2f}",
                    f"${balance_stats['max']:,.2f}",
                    f"${balance_stats['mean']:,.2f}",
                    f"${balance_stats['50%']:,.2f}",
                    f"${balance_stats['std']:,.2f}"
                ]
            })
            st.table(stats_df)
        with col2:
            st.markdown("**Estadísticas de Drawdown**")
            dd_stats = df["Max Drawdown"].describe()
            dd_stats_df = pd.DataFrame({
                "Métrica": ["Mínimo", "Máximo", "Promedio", "Mediana", "Desv. Estándar"],
                "Valor": [
                    f"${dd_stats['min']:,.2f}", f"${dd_stats['max']:,.2f}",
                    f"${dd_stats['mean']:,.2f}", f"${dd_stats['50%']:,.2f}",
                    f"${dd_stats['std']:,.2f}"
                ]
            })
            st.table(dd_stats_df)

        st.markdown("---")
        st.markdown("### 📈 Percentiles de Rendimiento")
        percentiles = [10, 25, 50, 75, 90]
        balance_percentiles = np.percentile(df["Balance Final"], percentiles)
        fig_percentiles = go.Figure()
        fig_percentiles.add_trace(go.Bar(x=[f"P{p}" for p in percentiles], y=balance_percentiles,
                                       marker_color=["#DC3545", "#FFC107", "#2E86AB", "#28A745", "#28A745"],
                                       text=[f"${v:,.0f}" for v in balance_percentiles], textposition="outside"))
        fig_percentiles.add_hline(y=config["target_pass"], line_dash="dash", line_color="#28A745",
                                annotation_text="Target Pass")
        fig_percentiles.add_hline(y=config["balance_inicial"], line_dash="dash", line_color="#999",
                                annotation_text="Balance Inicial")
        fig_percentiles.update_layout(title="Percentiles de Balance Final", xaxis_title="Percentil",
                                    yaxis_title="Balance ($)", height=400)
        st.plotly_chart(fig_percentiles, use_container_width=True)
    else:
        st.info("👆 Primero ejecuta una simulación en la pestaña 'Evaluación Principal' para ver el análisis avanzado")



st.sidebar.markdown("---")
st.sidebar.markdown(f"**Versión:** {__version__}")
st.sidebar.markdown("**Autor:** De Crypto Club")

st.sidebar.markdown("**GitHub:** [de-crypto-club](https://github.com/de-crypto-club)")
