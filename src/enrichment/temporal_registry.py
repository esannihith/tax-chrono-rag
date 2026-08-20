import re
from typing import Dict, Any, Optional


STATUTORY_EFFECTIVE_DATE_REGISTRY: Dict[str, Dict[str, Any]] = {
    '3': {
        'rule_title': 'Valuation of Perquisites (Accommodation, Motor Cars, Loans, Food)',
        'standard_commencement': '1962-04-01',
        'earliest_applicable_ay': 1962,
        'earliest_applicable_ay_str': 'AY 1962-63',
        'amendments': [
            {
                'notification_no': 'Notification No. 65/2023',
                'effective_date': '2023-09-01',
                'effective_ay': 2024,
                'effective_ay_str': 'AY 2024-25',
                'subject': 'Revised Rent-Free Accommodation rates (10%/7.5%/5% and 10% leased cap) based on 2011 census population tiers (>40L, 15L-40L, <=15L)',
                'prior_provision': 'Historical rates (15%/10%/7.5% and 15% leased cap) based on 2001 census population tiers (>25L, 10L-25L, <=10L)'
            }
        ],
        'notes': 'Rule 3 is in force continuously under the 1962 Rules. Notification No. 65/2023 revised accommodation perquisite rates w.e.f. 01-09-2023 (AY 2024-25 onwards).'
    },
    '12AC': {
        'rule_title': 'Updated Return of Income (Form ITR-U)',
        'inserted_date': '2022-04-29',
        'notification_no': 'Notification No. 48/2022',
        'earliest_applicable_ay': 2022,
        'earliest_applicable_ay_str': 'AY 2022-23',
        'notes': 'Rule 12AC was inserted w.e.f. 29-04-2022 via Notification No. 48/2022. It is legally not available for assessment years prior to AY 2022-23.'
    },
    '26D': {
        'rule_title': 'Declaration for Specified Senior Citizen Pensioners (Form 12BBA u/s 194P)',
        'inserted_date': '2021-09-02',
        'notification_no': 'Notification No. 98/2021',
        'earliest_applicable_ay': 2021,
        'earliest_applicable_ay_str': 'AY 2021-22',
        'notes': 'Rule 26D and Form 12BBA were inserted w.e.f. 02-09-2021 (Finance Act 2021).'
    },
    '21AAA': {
        'rule_title': 'Taxation of Relief from Specified Foreign Retirement Funds (Section 89A)',
        'inserted_date': '2022-04-04',
        'notification_no': 'Notification No. 24/2022',
        'earliest_applicable_ay': 2022,
        'earliest_applicable_ay_str': 'AY 2022-23',
        'notes': 'Rule 21AAA was inserted w.e.f. 04-04-2022 via Notification No. 24/2022.'
    },
    '114AAA': {
        'rule_title': 'Manner of making PAN inoperative upon non-linking with Aadhaar',
        'inserted_date': '2020-02-13',
        'notification_no': 'Notification No. 11/2020',
        'earliest_applicable_ay': 2020,
        'earliest_applicable_ay_str': 'AY 2020-21',
        'notes': 'Rule 114AAA was inserted w.e.f. 13-02-2020 via Notification No. 11/2020.'
    },
    '2BB': {
        'rule_title': 'Prescribed Allowances under Section 10(14)',
        'standard_commencement': '1962-04-01',
        'earliest_applicable_ay': 1962,
        'earliest_applicable_ay_str': 'AY 1962-63',
        'notes': 'Rule 2BB specifies exempt allowances including HRA, travel, uniform, and transport allowance for differently-abled employees.'
    },
    '31': {
        'rule_title': 'Certificate of Tax Deducted at Source (Form 16 / Form 16A)',
        'standard_commencement': '1962-04-01',
        'earliest_applicable_ay': 1962,
        'earliest_applicable_ay_str': 'AY 1962-63',
        'notes': 'Rule 31 prescribes TDS certificates to be issued by deductors to deductees.'
    }
}


class StatutoryTemporalRegistry:
    """Authoritative statutory registry validating rule effective dates and amendment applicability."""

    @classmethod
    def verify_rule_effective_date(
        cls,
        rule_id: str,
        date_or_ay: str,
        regime: str = '1962'
    ) -> Dict[str, Any]:
        clean_id = rule_id.upper().replace('RULE', '').replace('SEC', '').strip()
        target_year = '2026' if '2026' in regime else '1962'

        if target_year == '2026':
            return {
                'rule_id': clean_id,
                'regime': '2026',
                'queried_period': date_or_ay,
                'is_in_force_for_period': False,
                'status': 'draft_proposal',
                'effective_date_recorded': 'Draft statutory commencement',
                'scope_note': 'Income-tax Rules, 2026 are currently in draft status and have not been officially gazetted.'
            }

        eff_record = STATUTORY_EFFECTIVE_DATE_REGISTRY.get(clean_id)

        # Extract 4-digit year from date_or_ay
        year_match = re.search(r'\b(\d{4})\b', date_or_ay)
        queried_year = int(year_match.group(1)) if year_match else None

        if eff_record is None:
            return {
                'rule_id': clean_id,
                'regime': '1962',
                'queried_period': date_or_ay,
                'is_in_force_for_period': True,
                'status': 'standard_commencement',
                'effective_date_recorded': '1962-04-01',
                'scope_note': f'Rule {clean_id} is part of the Income-tax Rules, 1962 and in force during queried period.'
            }

        earliest_ay = eff_record.get('earliest_applicable_ay', 1962)
        in_force = True
        scope_note = eff_record.get('notes', f'Rule {clean_id} is in force during queried period.')

        if queried_year is not None and queried_year < earliest_ay:
            in_force = False
            scope_note = (
                f"Rule {clean_id} ({eff_record['rule_title']}) was inserted w.e.f. "
                f"{eff_record.get('inserted_date', 'subsequent notification')} via "
                f"{eff_record.get('notification_no', 'statutory amendment')} and was "
                f"NOT in force for periods prior to {eff_record['earliest_applicable_ay_str']}."
            )
        elif 'amendments' in eff_record and queried_year is not None:
            # Check specific amendment boundaries (e.g. Rule 3 accommodation rates)
            for amd in eff_record['amendments']:
                if queried_year < amd['effective_ay']:
                    scope_note = (
                        f"Rule {clean_id} was in force for {date_or_ay}, but governed by {amd['prior_provision']}. "
                        f"{amd['notification_no']} took effect later w.e.f. {amd['effective_ay_str']}."
                    )
                else:
                    scope_note = (
                        f"Rule {clean_id} is in force for {date_or_ay} under amended provisions "
                        f"({amd['notification_no']} w.e.f. {amd['effective_ay_str']}: {amd['subject']})."
                    )

        eff_date_str = eff_record.get('inserted_date', eff_record.get('standard_commencement', '1962-04-01'))

        return {
            'rule_id': clean_id,
            'regime': '1962',
            'queried_period': date_or_ay,
            'is_in_force_for_period': in_force,
            'status': 'in_force' if in_force else 'not_in_force',
            'effective_date_recorded': eff_date_str,
            'scope_note': scope_note
        }
